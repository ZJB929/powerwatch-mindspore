#!/usr/bin/env python3
"""Generate deterministic 15-minute household power data with labeled anomalies."""

from __future__ import annotations

import csv
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path


def generate(path: Path, days: int = 30, seed: int = 20260917) -> None:
    rng = random.Random(seed)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    points = days * 96
    anomaly_ranges = [(18 * 96 + 24, 18 * 96 + 40), (25 * 96 + 72, 25 * 96 + 84)]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "power_kw", "label"])
        for index in range(points):
            moment = start + timedelta(minutes=15 * index)
            slot = index % 96
            hour = slot / 4.0
            base = 0.18
            morning = 0.8 * math.exp(-((hour - 7.5) / 1.3) ** 2)
            evening = 1.2 * math.exp(-((hour - 19.0) / 2.0) ** 2)
            value = max(0.05, base + morning + evening + rng.gauss(0, 0.035))
            label = 0
            for left, right in anomaly_ranges:
                if left <= index < right:
                    value += 1.8
                    label = 1
            writer.writerow([moment.isoformat(), f"{value:.6f}", label])


if __name__ == "__main__":
    generate(Path("demo_power.csv"))
    print("generated demo_power.csv")
