"""Training loop for Conditional Flow Matching."""
import sys
import time
from pathlib import Path

import torch
from torch.optim import Adam
from tqdm import tqdm

from rafm.flow_matching.loss import cfm_loss
from rafm.utils.io import get_run_dir, save_checkpoint, save_config
from rafm.utils.logging import RunLogger
from rafm.utils.seeds import set_all_seeds


class Trainer:
    """CFM trainer.

    Args:
        model:   MLP
        path:    EuclideanPath or SphericalGeodesicPath
        source:  source distribution object
        dataset: dataset object with sample_train(n) and sample_val(n) methods
        cfg:     config dict (loaded from YAML)
        seed:    random seed for this run
        run_dir: output directory for this run
    """

    def __init__(self, model, path, source, dataset, cfg: dict, seed: int, run_dir: Path):
        self.model = model
        self.path = path
        self.source = source
        self.dataset = dataset
        self.cfg = cfg
        self.seed = seed
        self.run_dir = Path(run_dir)
        self.device = cfg.get("device", "cpu")
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = self.model.to(self.device)
        # torch.compile: skip on Windows (no Triton), use on Linux
        if self.device == "cuda" and hasattr(torch, "compile") and sys.platform != "win32":
            self.model = torch.compile(self.model)
        self.optimizer = Adam(self.model.parameters(), lr=cfg.get("lr", 1e-3))
        self.logger = RunLogger(self.run_dir, cfg, seed)

        # Pre-load training data on GPU to avoid CPU->GPU transfer each step
        self._train_data_gpu = dataset.get_train_data().to(self.device)

        # TensorBoard
        self._tb_writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._tb_writer = SummaryWriter(log_dir=str(self.run_dir / "tb"))
        except ImportError:
            pass

        save_config(cfg, self.run_dir / "config.yaml")

    def train(self) -> dict:
        """Run full training loop.

        Returns:
            dict with final training stats
        """
        set_all_seeds(self.seed)
        cfg = self.cfg
        n_steps = cfg.get("n_train_steps", 100_000)
        batch_size = cfg.get("batch_size", 256)
        log_every = cfg.get("log_every", 500)
        ckpt_every = cfg.get("ckpt_every", 10_000)

        self.model.train()
        t0 = time.time()

        n_train = self._train_data_gpu.shape[0]

        pbar = tqdm(range(1, n_steps + 1), desc=str(self.run_dir.name), dynamic_ncols=True)
        for step in pbar:
            idx = torch.randint(n_train, (batch_size,), device=self.device)
            x1 = self._train_data_gpu[idx]

            self.optimizer.zero_grad(set_to_none=True)
            loss = cfm_loss(self.model, self.path, self.source, x1, device=self.device)
            loss.backward()
            self.optimizer.step()

            if step % log_every == 0:
                loss_val = loss.item()
                elapsed = time.time() - t0
                pbar.set_postfix(loss=f"{loss_val:.4f}", elapsed=f"{elapsed:.0f}s")
                self.logger.log_step({
                    "step": step,
                    "loss": loss_val,
                    "elapsed_s": elapsed,
                })
                if self._tb_writer:
                    self._tb_writer.add_scalar("loss", loss_val, step)

            if step % ckpt_every == 0:
                save_checkpoint(
                    self.model, self.optimizer, step,
                    self.run_dir / "checkpoint.pt"
                )

        pbar.close()
        if self._tb_writer:
            self._tb_writer.close()
        self.logger.close()
        total_time = time.time() - t0
        stats = {"total_train_time_s": total_time, "final_loss": loss.item()}
        return stats
