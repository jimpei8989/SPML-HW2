import json
from pathlib import Path

from omegaconf import OmegaConf

import torch


class Recorder:
    def __init__(self, root_dir: Path, root_cfg) -> None:
        self.root_dir = root_dir

        self.root_dir.mkdir(parents=True, exist_ok=True)

        OmegaConf.save(root_cfg, self.root_dir / "config.yaml")

        self.attacking_log = []
        self.training_log = []

    def on_attack_ends(self, epoch, **kwargs):
        kwargs["epoch"] = epoch
        self.attacking_log.append(kwargs)

    def on_epoch_ends(self, epoch, **kwargs):
        kwargs["epoch"] = epoch
        self.training_log.append(kwargs)

    def save_checkpoint(
        self, epoch, model=None, optimizer=None, lr_scheduler=None, scheduler_type=None
    ):
        self.dump_logs()
        checkpoint = {
            "model_state_dict": model.state_dict() if model else None,
            "optimizer_state_dict": optimizer.state_dict() if model else None,
            "scheduler_state_dict": lr_scheduler.state_dict() if model else None,
            "scheduler_type": scheduler_type,
        }

        torch.save(checkpoint, self.root_dir / f"epoch_{epoch:03d}.pt")

    def dump_logs(self):
        with open(self.root_dir / "training_log.json", "w") as f:
            json.dump(self.training_log, f, indent=2)

        with open(self.root_dir / "attacking_log.json", "w") as f:
            json.dump(self.attacking_log, f, indent=2)
