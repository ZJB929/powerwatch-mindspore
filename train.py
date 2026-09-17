#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import mindspore as ms
from mindspore import Tensor, nn, save_checkpoint

from data import fit_scaler, load_series, make_windows, split_points


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


def reconstruction_error(model: nn.Cell, windows: np.ndarray) -> np.ndarray:
    prediction = model(Tensor(windows, ms.float32)).asnumpy()
    return np.mean((prediction - windows) ** 2, axis=1)


def metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float | int]:
    predicted = scores > threshold
    positive = labels == 1
    tp = int(np.logical_and(predicted, positive).sum())
    fp = int(np.logical_and(predicted, ~positive).sum())
    fn = int(np.logical_and(~predicted, positive).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall}


def train(args) -> dict:
    ms.set_seed(args.seed)
    np.random.seed(args.seed)
    series = load_series(args.csv)
    train_end, valid_end = split_points(len(series.values))
    mean, std = fit_scaler(series.values[:train_end])

    train_x, train_y, _ = make_windows(
        series.values[:train_end], series.labels[:train_end], args.window, args.stride, mean, std
    )
    valid_x, valid_y, _ = make_windows(
        series.values[train_end:valid_end],
        series.labels[train_end:valid_end],
        args.window,
        args.stride,
        mean,
        std,
    )
    test_x, test_y, test_ends = make_windows(
        series.values[valid_end:], series.labels[valid_end:], args.window, args.stride, mean, std
    )

    clean_train = train_x[train_y == 0]
    clean_valid = valid_x[valid_y == 0]
    if len(clean_train) == 0 or len(clean_valid) == 0 or len(test_x) == 0:
        raise ValueError("not enough windows after chronological split")

    model = AutoEncoder(args.window)
    loss_fn = nn.MSELoss()
    optimizer = nn.Adam(model.trainable_params(), learning_rate=args.learning_rate)
    train_step = nn.TrainOneStepCell(nn.WithLossCell(model, loss_fn), optimizer)
    train_step.set_train()

    rng = np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        order = rng.permutation(len(clean_train))
        losses = []
        for start in range(0, len(order), args.batch_size):
            batch = clean_train[order[start : start + args.batch_size]]
            loss = train_step(Tensor(batch, ms.float32), Tensor(batch, ms.float32))
            losses.append(float(loss.asnumpy()))
        print(f"epoch={epoch + 1:03d} loss={np.mean(losses):.7f}")

    model.set_train(False)
    valid_scores = reconstruction_error(model, clean_valid)
    threshold = float(np.quantile(valid_scores, args.quantile))
    test_scores = reconstruction_error(model, test_x)
    report = metrics(test_scores, test_y, threshold)
    report.update(
        {
            "threshold": threshold,
            "mean": mean,
            "std": std,
            "window": args.window,
            "stride": args.stride,
            "train_windows": int(len(clean_train)),
            "valid_windows": int(len(clean_valid)),
            "test_windows": int(len(test_x)),
        }
    )

    args.output.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, str(args.output / "powerwatch.ckpt"))
    (args.output / "metadata.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    anomaly_indices = np.where(test_scores > threshold)[0]
    print(json.dumps(report, ensure_ascii=False, indent=2))
    for idx in anomaly_indices[:20]:
        point = valid_end + int(test_ends[idx])
        print(f"alert timestamp={series.timestamps[point]} score={test_scores[idx]:.7f}")
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="MindSpore household power anomaly detector")
    result.add_argument("--csv", type=Path, required=True)
    result.add_argument("--output", type=Path, default=Path("artifacts"))
    result.add_argument("--window", type=int, default=96, help="points per window")
    result.add_argument("--stride", type=int, default=4)
    result.add_argument("--epochs", type=int, default=40)
    result.add_argument("--batch-size", type=int, default=32)
    result.add_argument("--learning-rate", type=float, default=1e-3)
    result.add_argument("--quantile", type=float, default=0.995)
    result.add_argument("--seed", type=int, default=20260917)
    return result


if __name__ == "__main__":
    train(parser().parse_args())
