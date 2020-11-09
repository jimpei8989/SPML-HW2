from pathlib import Path

from omegaconf import OmegaConf

import torch


class Recorder:
    def __init__(self, root_dir: Path, cfg) -> None:
        self.root_dir = root_dir

        if not self.root_dir.is_dir():
            self.root_dir.mkdir()

        self.cfg = cfg

        OmegaConf.save(self.cfg, self.root_dir / "config.yaml")

    def on_epoch_ends(self):
        pass

    def save_checkpoint(self, epoch, model=None, optimizer=None, lr_scheduler=None):
        checkpoint = {
            "model_state_dict": model.state_dict() if not model else None,
            "optimizer_state_dict": optimizer.state_dict() if not model else None,
            "lr_scheduler_state_dict": lr_scheduler.state_dict() if not model else None,
        }

        torch.save(checkpoint, self.root_dir / f"epoch_{epoch:03d}.pt")
