import torch
from torch import nn


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        mean, std = map(torch.as_tensor, [mean, std])
        self.mean = nn.Parameter(mean.view(3, 1, 1), requires_grad=False)
        self.std = nn.Parameter(std.view(3, 1, 1), requires_grad=False)

    def forward(self, x):
        return (x - self.mean) / self.std
