# PowerWatch-MindSpore

使用 MindSpore 一维自编码器检测家庭用电曲线异常。项目自带固定随机种子的合成数据发生器，不依赖可能失效的外部数据集链接。

## 工程原则

- 训练、验证、测试集严格按时间切分；
- 标准化统计量只使用训练段；
- 自编码器只学习正常训练窗口；
- 告警阈值只使用验证段中的正常窗口校准；
- 测试段仅用于最终评估；
- 保存模型之外，同时保存均值、标准差、窗口长度和阈值。

## 快速开始

先按 MindSpore 官方指南安装与 CPU 或昇腾环境匹配的版本，然后执行：

```bash
python3 generate_demo_data.py
python3 train.py --csv demo_power.csv --epochs 40
```

纯数据管道测试不依赖 MindSpore：

```bash
python3 -m unittest -v test_data.py
```

训练产物：

```text
artifacts/
├── powerwatch.ckpt
└── metadata.json
```

## 重要限制

本项目输出的是“偏离历史模式”的软件告警，不是电气故障诊断，不能替代断路器、漏电保护器和专业巡检，也不应在没有人工确认及独立安全设计的情况下直接控制断电设备。

当前生成环境未安装 MindSpore，因此仓库不附带虚构训练指标。复现者应公开真实版本、设备、随机种子以及多次运行的均值和标准差。

完整数据泄漏分析、阈值设计和端侧路线见 [技术文章](docs/article.md)。

## License

Apache-2.0
