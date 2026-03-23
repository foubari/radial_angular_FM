"""Structured run logging: JSON metadata + CSV metrics."""
import csv
import json
import time
from pathlib import Path


class RunLogger:
    def __init__(self, run_dir: Path, config: dict, seed: int):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.run_dir / "train_log.csv"
        self._csv_writer = None
        self._csv_file = None

        meta = {
            "config": config,
            "seed": seed,
            "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        (self.run_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    def _get_writer(self, fieldnames: list[str]):
        if self._csv_writer is None:
            self._csv_file = open(self.csv_path, "w", newline="")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames)
            self._csv_writer.writeheader()
        return self._csv_writer

    def log_step(self, row: dict) -> None:
        writer = self._get_writer(list(row.keys()))
        writer.writerow(row)
        if self._csv_file:
            self._csv_file.flush()

    def save_metrics(self, metrics: dict) -> None:
        (self.run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    def close(self) -> None:
        if self._csv_file:
            self._csv_file.close()
