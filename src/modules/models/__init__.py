import random

import torch
from torch import nn, optim
from pytorchcv.model_provider import get_model

from .normalize import Normalize


class CIFAR10_Model(nn.Module):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    def __init__(self, cfg, weight_path=None):
        super().__init__()
        self.num_epochs = cfg.get("num_epochs", 0)

        self.model_names = cfg.models if cfg.name == "Ensemble" else [cfg.name]

        for name in self.model_names:
            self.add_module(
                name,
                nn.Sequential(
                    Normalize(self.mean, self.std),
                    get_model(name + "_cifar10", pretrained=True),
                ),
            )

        if weight_path:
            ckpt = torch.load(weight_path)
            if "model_state_dict" in ckpt:
                self.load_state_dict(ckpt["model_state_dict"])
            else:
                self.load_state_dict(ckpt)

    def get_single_model(self):
        return random.choice(self.model_names)

    def forward(self, x):
        return torch.mean(torch.stack([m(x) for m in self.children()], dim=1), dim=1)

    def dump_weights(self, path):
        self.cpu()
        torch.save(self.state_dict(), path)


def build_optimizer(cfg, parameters):
    cfg = cfg.copy()
    name = cfg.pop("name")

    scheduler_cfg = cfg.pop("scheduler") if "scheduler" in cfg else None

    if hasattr(optim, name):
        optimizer = getattr(optim, name)(parameters, **cfg)
    else:
        raise NotImplementedError("Optimizer not implemented")

    return optimizer, build_scheduler(scheduler_cfg, optimizer)


def build_scheduler(cfg, optimizer):
    cfg = cfg.copy()
    name = cfg.pop("name")

    step_type = cfg.pop("type")

    if hasattr(optim.lr_scheduler, name):
        scheduler = getattr(optim.lr_scheduler, name)(optimizer, **cfg)
    else:
        raise NotImplementedError("Scheduler not implemented")

    return scheduler, step_type
