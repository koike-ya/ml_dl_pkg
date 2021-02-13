from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import random
import torch
from torch import Tensor
from torchaudio.transforms import MelSpectrogram, TimeMasking, FrequencyMasking, ComputeDeltas, TimeStretch, \
                                  AmplitudeToDB
import numpy as np


@dataclass
class AugConfig:
    n_time_mask: int = 1  # maximum possible number of the mask.
    time_mask_len: int = 10  # maximum possible length of the mask. Indices uniformly sampled from [0, time_mask_param)
    n_freq_mask: int = 1  # maximum possible number of the mask.
    freq_mask_len: int = 10  # maximum possible length of the mask. Indices uniformly sampled from [0, freq_mask_param).
    mask_value: float = 1e-2  # Maximum possible value assigned to the masked columns.
    spec_aug_prob: float = 0.5  # Probability of SpecAugment.
    trim_sec: float = 5  # Trim audio segment with speficied length, pad if shorter.
    amp_change_scale: List[float] = field(default_factory=lambda: [0.2, 5])  # Randomly Change amplitude of signal
    amp_change_prob: float = 0.5   # Probability of randomly change amplitude of signal


class TimeFreqMask(TimeMasking, FrequencyMasking):
    axes = ['time', 'freq']

    def __init__(self, p: float = 0.5, max_time_mask_idx: int = 1, max_mask_value: float = 0,
                 max_n_mask: int = 1, axis='time') -> None:
        assert axis in TimeFreqMask.axes
        self.p = p
        self.max_mask_value = max_mask_value
        self.max_n_mask = max_n_mask
        if axis == 'time':
            super(TimeMasking, self).__init__(max_time_mask_idx, False)
        else:
            super(FrequencyMasking, self).__init__(max_time_mask_idx, 1, False)

    def forward(self, specgram: Tensor) -> Tensor:
        if random.uniform(0, 1) < self.p:
            for _ in range(random.randint(1, self.max_n_mask)):
                mask_value = random.uniform(0, 1) * self.max_mask_value
                specgram = super().forward(specgram, mask_value)
            return specgram
        return specgram


class Trim(torch.nn.Module):
    def __init__(self, sr, trim_sec, randomly=False):
        super(Trim, self).__init__()
        self.trim_idxs = int(sr * trim_sec)
        self.randomly = randomly

    def forward(self, x: Tensor):
        if x.size(0) < self.trim_idxs:
            len_pad = self.trim_idxs - x.size(0)
            x = torch.from_numpy(np.pad(x, (len_pad - len_pad // 2, len_pad // 2), mode='reflect'))

        if self.randomly:
            start_idx = torch.randint(high=x.size(0) - self.trim_idxs, size=(1,)).item()
        else:
            start_idx = (x.size(0) - self.trim_idxs) // 2
        return x[start_idx:start_idx + self.trim_idxs]


class RandomAmpChange(torch.nn.Module):
    def __init__(self, p=0.5, scale=(0.2, 5.0)):
        super(RandomAmpChange, self).__init__()
        self.p = p
        self.scale = scale

    def forward(self, x: Tensor):
        if random.uniform(0, 1) < self.p:
            return x * random.uniform(self.scale[0], self.scale[1])
        return x
