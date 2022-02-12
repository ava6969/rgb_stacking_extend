import numpy as np
import torch
import torch.nn as nn
from a2c_ppo_acktr.a2c_ppo_acktr.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Sum(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.sum(x, self.dim)


class Mean(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.mean(x, self.dim)


def init_rec(rec):
    for name, param in rec.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param)
    return rec


def init_(m):
    return init(m, nn.init.orthogonal_, lambda x: nn.init.
                constant_(x, 0), np.sqrt(2))


