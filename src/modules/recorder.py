import json
from pathlib import Path

from omegaconf import OmegaConf

import torch


class Recorder:
    def __init__(self, root_dir: Path, cfg) -> None:
        self.root_dir = root_dir

        if not self.root_dir.is_dir():
            self.root_dir.mkdir()

        OmegaConf.save(cfg, self.root_dir / "config.yaml")

        self.training_log = []

    def on_epoch_ends(self, **kwargs):
        kwargs["epoch"] = len(self.training_log)
        self.training_log.append(kwargs)

    def save_checkpoint(self, epoch, model=None, optimizer=None, lr_scheduler=None):
        checkpoint = {
            "model_state_dict": model.state_dict() if not model else None,
            "optimizer_state_dict": optimizer.state_dict() if not model else None,
            "lr_scheduler_state_dict": lr_scheduler.state_dict() if not model else None,
        }

        torch.save(checkpoint, self.root_dir / f"epoch_{epoch:03d}.pt")

    def finish_training(self, model):
        with open(self.root_dir / "training_log.json", "w") as f:
            json.dump(self.training_log, f, indent=2)
        torch.save(model.state_dict(), self.root_dir / "model_weights.pt")
