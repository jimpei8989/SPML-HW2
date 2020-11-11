import torch.optim
from pytorchcv.model_provider import get_model


def build_model(name):
    return get_model(name + "_cifar10", pretrained=True)


def build_optimizer(cfg, parameters):
    cfg = cfg.copy()
    name = cfg.pop("name")

    scheduler_cfg = cfg.pop("scheduler") if "scheduler" in cfg else None

    if hasattr(torch.optim, name):
        optimizer = getattr(torch.optim, name)(parameters, **cfg)
    else:
        raise NotImplementedError("Optimizer not implemented")

    return optimizer, build_scheduler(scheduler_cfg, optimizer)


def build_scheduler(cfg, optimizer):
    cfg = cfg.copy()
    name = cfg.pop("name")

    step_type = cfg.pop("type")

    if hasattr(torch.optim.lr_scheduler, name):
        scheduler = getattr(torch.optim.lr_scheduler, name)(optimizer, **cfg)
    else:
        raise NotImplementedError("Scheduler not implemented")

    return scheduler, step_type
