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

    white_p: float = 0.5  # Probability of white noise
    sigma: float = 1.0  # sigma of gaussian distribution of white noise

    stretch_p: float = 1.0  # Probability of time stretch
    stretch_range: Tuple[float, float] = (0.8, 1.2)  # Time Stretch speedup/slow down rate


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


class DynamicTimeStretch(TimeStretch):
    def __init__(self, p, hop_length, n_freq, stretch_range) -> None:
        self.p = p
        self.stretch_range = stretch_range
        super(DynamicTimeStretch, self).__init__(hop_length, n_freq, None)

    def forward(self, specgram: Tensor) -> Tensor:
        if random.uniform(0, 1) < self.p:
            orig_freq, orig_time, _ = list(specgram.size())
            stretch_value = random.uniform(*self.stretch_range)
            specgram = super().forward(specgram, stretch_value)
            specgram = specgram.pow(2.).sum(-1)

            if stretch_value > 1.0:
                len_pad = orig_time - specgram.size(1)
                before = torch.zeros(orig_freq, len_pad - len_pad // 2)
                after = torch.zeros(orig_freq, len_pad // 2)
                specgram = torch.cat([before, specgram, after], dim=1)
            elif stretch_value < 1.0:
                start_idx = (specgram.size(1) - orig_time) // 2
                specgram = specgram[:, start_idx:start_idx + orig_time]

            assert (orig_time == specgram.size(1)) and (orig_freq == specgram.size(0)), (orig_freq, orig_time, specgram.size())
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
            start_idx = torch.randint(high=max(1, x.size(0) - self.trim_idxs), size=(1,)).item()
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


class WhiteNoise(torch.nn.Module):
    def __init__(self, p=0.5, sigma=1.0):
        super(WhiteNoise, self).__init__()
        self.p = p
        self.sigma = sigma

    def forward(self, x: Tensor):
        if random.uniform(0, 1) < self.p:
            return x + torch.normal(0.0, self.sigma, x.size())
        return x
