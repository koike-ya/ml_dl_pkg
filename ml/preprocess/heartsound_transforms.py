from dataclasses import dataclass
from typing import List, Dict, Tuple
import random
import numpy as np

import torch
from torch import Tensor


@dataclass
class HSTransConfig:
    resp_p: float = 0.5
    resp_period: float = 1.0
    resp_amp_scale: Tuple[float] = (0.2, 5.0)


class RespScale(torch.nn.Module):
    def __init__(self, p: float = 0.5, sr: int = 2000, resp_period: float = 1.0,
                 resp_amp_scale: Tuple[float] = (0.2, 5.0)) -> None:
        super(RespScale, self).__init__()
        self.p = p
        self.sr = sr
        self.resp_period = resp_period
        self.resp_amp_scale = resp_amp_scale

    def forward(self, x: Tensor):
        if random.uniform(0, 1) < self.p:
            phase = random.uniform(0, 2 * np.pi)
            amp_scale = (random.uniform(self.resp_amp_scale[0], 1.0), random.uniform(1.0, self.resp_amp_scale[1]))
            t = np.linspace(phase, np.pi * 2 * len(x) / self.sr * self.resp_period + phase, len(x))
            weight = (-np.cos(t) + 1) / 2
            weight *= amp_scale[1] - amp_scale[0]
            weight += amp_scale[0]
            return x * torch.from_numpy(weight).to(dtype=torch.float)
        return x
