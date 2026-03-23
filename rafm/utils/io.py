"""Checkpointing, directory management, config I/O."""
import json
from pathlib import Path

import torch
import yaml


def get_run_dir(base: str, experiment: str, dataset: str, method: str, seed: int) -> Path:
    return Path(base) / experiment / dataset / method / f"seed_{seed}"


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    step: int, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)


def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer | None,
                    path: Path) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["step"]


def save_samples(samples: torch.Tensor, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(samples, path)


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # Handle inheritance
    if "inherit" in cfg:
        base_path = Path(path).parent.parent / cfg.pop("inherit")
        base = load_config(base_path)
        base.update(cfg)
        return base
    return cfg


def save_config(cfg: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
