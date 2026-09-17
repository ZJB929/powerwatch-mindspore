import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data import fit_scaler, load_series, make_windows, split_points


class DataTests(unittest.TestCase):
    def test_window_label_covers_whole_window(self):
        values = np.arange(10, dtype=np.float32)
        labels = np.zeros(10, dtype=np.int32)
        labels[3] = 1
        windows, window_labels, ends = make_windows(values, labels, 4, 2, 0.0, 1.0)
        self.assertEqual(windows.shape, (4, 4))
        self.assertEqual(window_labels.tolist(), [1, 1, 0, 0])
        self.assertEqual(ends.tolist(), [3, 5, 7, 9])

    def test_scaler_uses_train_only(self):
        train = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
        mean, std = fit_scaler(train)
        self.assertAlmostEqual(mean, 2.0)
        self.assertGreater(std, 0)

    def test_rejects_negative_power(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["timestamp", "power_kw", "label"])
                writer.writerow(["2026-01-01T00:00:00Z", -1.0, 0])
            with self.assertRaises(ValueError):
                load_series(path)

    def test_split_is_chronological_boundaries(self):
        self.assertEqual(split_points(100), (60, 80))

    def test_validation_filter_can_exclude_anomalous_windows(self):
        values = np.arange(12, dtype=np.float32)
        labels = np.zeros(12, dtype=np.int32)
        labels[5] = 1
        windows, window_labels, _ = make_windows(values, labels, 4, 2, 0.0, 1.0)
        clean = windows[window_labels == 0]
        self.assertEqual(len(windows), 5)
        self.assertEqual(len(clean), 3)


if __name__ == "__main__":
    unittest.main()
