import torch
from torch import nn, optim
from pytorchcv.model_provider import get_model

from modules.normalize import Normalize


cifar10_mean = [0.4914, 0.4822, 0.4465]
cifar10_std = [0.2023, 0.1994, 0.2010]


class CIFAR10_Model(nn.Module):
    def __init__(self, cfg, log_dir, load_weight=False):
        super().__init__()
        self.model = get_model(cfg.name + "_cifar10", pretrained=True)

        self.normalize = Normalize(mean=cifar10_mean, std=cifar10_std)

        if load_weight:
            self.load_state_dict(torch.load(log_dir / "model_weights.pt"))

    def forward(self, x):
        return self.model(self.normalize(x))


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
