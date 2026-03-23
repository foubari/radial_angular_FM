"""Efficiency metrics: wall-clock, memory, NFE."""
import time

import torch


def get_peak_memory_mb(device: str = "cuda") -> float:
    """Return peak GPU memory usage in MB since last reset."""
    if device == "cuda" and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 ** 2
    return 0.0


def reset_memory_counter(device: str = "cuda") -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


class Timer:
    """Simple wall-clock timer context manager."""
    def __enter__(self):
        self._t0 = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self._t0
