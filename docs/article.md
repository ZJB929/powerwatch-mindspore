# 用 MindSpore 做家庭用电异常检测：从时间窗、数据泄漏到可解释告警

> 项目：PowerWatch-MindSpore  
> 技术栈：MindSpore、NumPy、一维自编码器  
> 适用场景：智能插座/电表曲线中的长时间高负载、待机异常和行为漂移  
> 开源协议建议：Apache-2.0

## 1. “电量变大”不一定是异常

家庭用电有很强的时间规律：早晚存在峰值，夜间通常较低，周末和工作日也可能不同。如果简单设置“功率超过 1 kW 就告警”，烧水、做饭和使用空调都会产生大量误报。

PowerWatch 的思路是让模型学习一段正常用电曲线的形状，然后比较输入与重建结果。模型不需要提前知道“空调忘关”属于哪一类异常，只要当前曲线明显偏离正常分布，重建误差就会升高。

完整链路如下：

```text
15 分钟功率采样
  ↓ 缺失值/非法值检查
96 点滑动窗口（一天）
  ↓ 仅使用训练段统计量标准化
自编码器：96 → 48 → 12 → 48 → 96
  ↓
每个窗口的 MSE 重建误差
  ↓ 验证集 99.5% 分位数校准
异常分数 + 时间戳 + 阈值
```

配套源码位于 `powerwatch-mindspore/`。正式公开时，应把本文中的目录名替换成固定 tag 的 GitCode 链接。

## 2. 为什么选择自编码器

家庭真实异常往往很少，而且种类会变化。监督分类需要大量带标签的“忘关电器”“设备老化”“传感器故障”样本，不容易获得。

自编码器只用正常样本训练。编码器把 96 个采样点压缩到 12 维，解码器尝试恢复原曲线：

```python
class AutoEncoder(nn.Cell):
    def __init__(self, window: int):
        super().__init__()
        self.layers = nn.SequentialCell(
            nn.Dense(window, 48),
            nn.ReLU(),
            nn.Dense(48, 12),
            nn.ReLU(),
            nn.Dense(12, 48),
            nn.ReLU(),
            nn.Dense(48, window),
        )

    def construct(self, x):
        return self.layers(x)
```

它不是唯一选择。规律更复杂时可以换成 1D CNN、LSTM 或 Transformer；设备和数据很少时，移动中位数、MAD 等稳健统计方法可能更合适。本文选择小型全连接自编码器，是为了让数据切分、阈值和告警逻辑更容易被检查。

## 3. 第一个技术坑：时间序列不能随机切分

普通图片分类常把样本随机打乱后切分。时间序列若这样做，相邻窗口会同时进入训练集和测试集，模型等于提前见过测试曲线的一部分。

项目严格按时间切成三段：

```python
def split_points(length: int, train_ratio=0.6, valid_ratio=0.2):
    train_end = int(length * train_ratio)
    valid_end = int(length * (train_ratio + valid_ratio))
    return train_end, valid_end
```

```text
过去 60%：训练模型
随后 20%：选择告警阈值
最后 20%：只做最终评估
```

不能根据测试集效果反复修改阈值，否则测试集又参与了模型选择。需要继续迭代时，应保留一个从未用于决策的最终留出集，或者采用滚动时间验证。

## 4. 第二个技术坑：标准化也会泄漏未来信息

标准化通常写成：

```python
x = (x - mean) / std
```

问题在于 `mean` 和 `std` 从哪里来。如果使用整份数据计算，测试时期的整体用电水平已经泄漏给训练流程。项目只使用训练段：

```python
mean = float(train_values.mean())
std = max(float(train_values.std()), 1e-6)
```

然后把同一组 `mean/std` 应用于验证和测试段。部署时也必须保存这两个值，不能在每个新窗口上重新标准化，否则异常幅度会被自己抵消。

保存的 `metadata.json` 至少应包含：

```json
{
  "mean": "训练段均值",
  "std": "训练段标准差",
  "window": 96,
  "stride": 4,
  "threshold": "验证集校准值",
  "model_version": "Git commit 或 tag"
}
```

## 5. 窗口标签应该覆盖整个窗口

假设一天窗口中的第 20 个采样点开始出现异常。如果只取窗口最后一个点的标签，这个窗口可能仍被标为正常，评估会产生错误。

项目的窗口标签取整个窗口的最大值：

```python
for start in range(0, len(values) - window + 1, stride):
    end = start + window
    xs.append((values[start:end] - mean) / std)
    ys.append(int(labels[start:end].max()))
```

这代表“窗口内任一点异常，则窗口异常”。它适合安全告警，但会扩大异常区间。如果业务更关注事件开始时间，可以同时保留点级标签，在窗口告警后用逐点重建误差定位贡献最大的时刻。

## 6. 阈值不是模型自己学出来的

自编码器输出的是连续重建误差：

```python
prediction = model(Tensor(windows, ms.float32)).asnumpy()
scores = np.mean((prediction - windows) ** 2, axis=1)
```

要转换成告警，需要一个阈值。项目使用验证集正常误差的 99.5% 分位数：

```python
threshold = float(np.quantile(valid_scores, 0.995))
predicted = test_scores > threshold
```

这不是普适最优值。分位数越高，误报通常越少，但可能漏掉轻微异常；分位数越低，召回率提高，告警疲劳也会加剧。

家庭场景更适合把阈值与连续窗口规则结合：

```text
单窗口超过阈值：记录，不推送
连续 3 个窗口超过阈值：低优先级提醒
高出阈值 5 倍且持续 30 分钟：高优先级提醒
```

此外应设置冷却时间，避免同一事件每 15 分钟重复通知。

## 7. 如何让告警可解释

只输出“异常分数 0.083”对普通用户没有意义。项目可以计算窗口内每个采样点的平方误差：

```python
point_error = (prediction - window) ** 2
peak = int(np.argmax(point_error))
```

再把 `peak` 映射回时间戳，生成更清楚的说明：

```text
2026-09-17 02:00—03:30 用电曲线偏离日常模式
最大偏差出现在 02:45
实际功率：2.14 kW
近 30 天相同时段中位数：0.21 kW
```

注意措辞应该是“偏离日常模式”，而不是“检测到电器故障”。单路总表无法判断具体设备，更不能代替电气安全检测。真正的设备归因需要分路计量、NILM 模型或用户反馈。

## 8. 没有数据集时，如何让项目仍可复现

很多教程引用一个以后可能失效的下载链接。PowerWatch 自带确定性的合成数据发生器：

```python
base = 0.18
morning = 0.8 * exp(-((hour - 7.5) / 1.3) ** 2)
evening = 1.2 * exp(-((hour - 19.0) / 2.0) ** 2)
value = base + morning + evening + gaussian_noise
```

然后在固定时间段注入持续高负载：

```python
if anomaly_start <= index < anomaly_end:
    value += 1.8
    label = 1
```

生成器使用固定随机种子，读者可以得到相同 CSV。它适合验证工程链路，不适合证明真实家庭环境的模型效果。文章必须把这两件事分开：

- 合成数据证明代码可复现、评估逻辑可检查；
- 真实数据才能证明项目对实际用电行为有价值。

公开真实家庭数据前，应移除住址、账户、设备标识等信息。即使只有功率曲线，也可能推断住户作息，因此不应未经同意上传。

## 9. MindSpore 训练实现

模型使用 `MSELoss` 和 Adam。为了让训练过程透明，项目直接使用 `TrainOneStepCell`：

```python
model = AutoEncoder(window=96)
loss_fn = nn.MSELoss()
optimizer = nn.Adam(model.trainable_params(), learning_rate=1e-3)
train_step = nn.TrainOneStepCell(nn.WithLossCell(model, loss_fn), optimizer)
train_step.set_train()

for epoch in range(epochs):
    for batch in batches:
        loss = train_step(
            Tensor(batch, ms.float32),
            Tensor(batch, ms.float32),
        )
```

训练数据只保留标签为正常的窗口：

```python
clean_train = train_x[train_y == 0]
```

真实部署通常没有完整标签，这时需要一个启动期：先采集若干周数据，结合稳健统计和人工检查清理明显异常，再训练基线模型。直接把所有历史数据视为正常，会让模型学会重建长期异常。

## 10. 复现步骤

先按 MindSpore 官方安装指南选择与 CPU 或昇腾环境匹配的版本。不要直接复制一个可能与当前驱动、CANN 或 Python 不兼容的固定安装命令。

```bash
python3 generate_demo_data.py
python3 train.py --csv demo_power.csv --epochs 40
```

训练完成后应得到：

```text
artifacts/
├── powerwatch.ckpt
└── metadata.json
```

为了让别人真正复现，还应保存：

- MindSpore 版本；
- Python 和 NumPy 版本；
- CPU/昇腾型号与 CANN 版本；
- 随机种子；
- Git commit；
- 输入 CSV 的 SHA-256；
- 窗口、步长、学习率、epoch、阈值分位数。

## 11. 评价指标不能只写 Accuracy

如果异常只占 1%，模型把所有窗口判断为正常，也有 99% Accuracy。更有意义的指标是：

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
```

- Precision 低：提醒太多，用户会关闭通知；
- Recall 低：真正的异常被漏掉；
- 事件级召回：一次持续异常只要成功提醒一次，可能就算命中；
- 每日误报数：比窗口级 Accuracy 更接近使用体验；
- 检测延迟：从异常开始到首次提醒花了多久。

建议发布如下表格，而不是只贴训练 loss：

| 数据 | Precision | Recall | 每日误报 | 中位检测延迟 |
|---|---:|---:|---:|---:|
| 合成数据 | **[实测]** | **[实测]** | **[实测]** | **[实测]** |
| 匿名真实数据 | **[实测]** | **[实测]** | **[实测]** | **[实测]** |

如果没有真实数据，就明确写“尚未验证”，不要把合成数据上的高分包装成落地效果。

## 12. 已完成的测试与待完成验证

项目的数据管道有五个可独立运行的单元测试：窗口标签覆盖、训练集独立标准化、负功率拒绝、时间切分边界，以及验证阈值排除异常窗口。

```text
test_rejects_negative_power ... ok
test_scaler_uses_train_only ... ok
test_split_is_chronological_boundaries ... ok
test_validation_filter_can_exclude_anomalous_windows ... ok
test_window_label_covers_whole_window ... ok

Ran 5 tests in 0.007s
OK
```

同时已对 `data.py`、数据生成器和训练脚本执行 Python 语法检查。

**当前生成环境没有安装 MindSpore，所以不能声称模型训练已经跑通。** 发布前至少需要补齐：

| 项目 | 结果 |
|---|---|
| MindSpore 版本与设备 | **[填写]** |
| 40 epoch 是否完成 | **[填写]** |
| 最终训练 loss | **[填写]** |
| 阈值 | **[填写]** |
| Precision / Recall | **[填写]** |
| 峰值内存、训练耗时 | **[填写]** |
| 固定种子重复 5 次均值与标准差 | **[填写]** |

只报告一次随机运行可能误导读者。样本较小时，建议至少重复 5 个种子，并给出均值和标准差。

## 13. 从实验走向端侧部署

项目后续可以沿华为开源生态继续扩展：

1. 在 MindSpore 中完成训练并冻结模型；
2. 转换为适合端侧推理的格式；
3. 在昇腾设备上用 CANN/AscendCL 执行推理；
4. 在 OpenHarmony 端显示趋势和告警；
5. 在 openEuler 家庭服务器保存历史数据和模型版本。

但每一步都应先回答“为什么需要”。如果一天只推理 96 个点，普通 CPU 已经足够；为了使用昇腾而强行增加部署复杂度没有工程价值。只有当需要多家庭、多传感器、更多模型或低延迟并发推理时，专用硬件才更有意义。

## 14. 安全边界

PowerWatch 不是电气保护装置，不能替代漏电保护器、断路器和专业巡检。它只根据历史曲线生成软件告警：

- 不能判断线路是否过热；
- 不能确认具体是哪台设备；
- 传感器离线时可能没有数据，而不是“功率为零”；
- 数据漂移后需要重新校准；
- 不能在未确认的情况下自动断电。

如果未来加入远程断电，必须把鉴权、重放防护、失败安全模式和人工确认作为独立安全设计，不能直接用模型输出控制继电器。

## 15. 小结

这类项目的含金量不在网络层数，而在于是否把评估链路做对：时间切分防止未来泄漏；标准化只看训练段；阈值由验证段校准；测试段不参与调参；告警输出可以回到具体时间；合成数据与真实效果不混为一谈。

把这些原则写进代码和测试后，PowerWatch 才是一个可以继续贡献的开源项目，而不只是又一个“loss 下降截图”。

## 参考资料

- [MindSpore 官方开源仓库](https://gitcode.com/mindspore/mindspore)
- [MindSpore 官方文档](https://www.mindspore.cn/docs/zh-CN/master/index.html)
- [NumPy quantile 文档](https://numpy.org/doc/stable/reference/generated/numpy.quantile.html)
- [NIST 关于 SHA 与数据完整性的资料入口](https://csrc.nist.gov/projects/hash-functions)
