import torch
from torch import nn
from pytorchcv.model_provider import get_model


class Ensemble(nn.Module):
    def __init__(self, models) -> None:
        super().__init__()
        for m in models:
            self.add_module(m, get_model(m + "_cifar10", pretrained=True))

    def forward(self, x):
        return torch.mean(torch.stack([m(x) for m in self.children()], dim=1), dim=1)
