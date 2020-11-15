import torch
from torch import nn, optim
from pytorchcv.model_provider import get_model

from .ensemble import Ensemble
from .normalize import Normalize


class CIFAR10_Model(nn.Module):
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2023, 0.1994, 0.2010)

    def __init__(self, cfg, weight_path=None, num_epochs=1):
        super().__init__()

        if cfg.name == "Ensemble":
            self.model = Ensemble(cfg.models)
        else:
            self.model = get_model(cfg.name + "_cifar10", pretrained=cfg.pretrained)
        self.normalize = Normalize(mean=self.cifar10_mean, std=self.cifar10_std)

        self.num_epochs = num_epochs

        if weight_path:
            ckpt = torch.load(weight_path)

            if "model_state_dict" in ckpt:
                self.load_state_dict(ckpt["model_state_dict"])
            else:
                self.load_state_dict(ckpt)

    def forward(self, x):
        return self.model(self.normalize(x))

    def dump_weights(self, path):
        self.cpu()
        torch.save(self.state_dict(), self.root_dir / "model_weights.pt")


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
