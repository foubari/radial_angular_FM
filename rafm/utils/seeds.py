import random
import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_seed_list(n_seeds: int, base_seed: int = 42) -> list[int]:
    rng = np.random.default_rng(base_seed)
    return rng.integers(0, 100_000, size=n_seeds).tolist()
