from dataclasses import dataclass, field
from typing import List, Dict

import torch
from torch import Tensor
from torchaudio.transforms import MelSpectrogram, TimeMasking, FrequencyMasking, ComputeDeltas, TimeStretch

from ml.utils.enums import TimeFrequencyFeature


@dataclass
class TransConfig:
    sample_rate: float = 500.0  # The sample rate for the data/model features
    transform_order: List[str] = field(default_factory=lambda: ['logmel', 'normalize'])
    n_fft: int = 800  # Size of FFT
    win_length: int = n_fft    # Window size for spectrogram in data points
    hop_length: int = n_fft // 2  # Window stride for spectrogram in data points
    n_mels: int = 64            # Number of mel filters banks
    transform: TimeFrequencyFeature = TimeFrequencyFeature.none
    f_min: float = 0.0  # High pass filter
    f_max: float = sample_rate / 2  # Low pass filter
    time_mask_len: int = 10  # maximum possible length of the mask. Indices uniformly sampled from [0, time_mask_param)
    freq_mask_len: int = 10  # maximum possible length of the mask. Indices uniformly sampled from [0, freq_mask_param).
    mask_value: float = 1e-2  # Maximum possible value assigned to the masked columns.
    delta: int = 5  # Compute delta coefficients of a tensor, usually a spectrogram
    stretch_rate: float = 1.0  # Time Stretch speedup/slow down rate


class TimeFreqMask(TimeMasking, FrequencyMasking):
    axes = ['time', 'freq']
    def __init__(self, max_time_mask_idx: int, max_mask_value: float, axis='time') -> None:
        assert axis in TimeFreqMask.axes
        self.max_mask_value = max_mask_value
        if axis == 'time':
            super(TimeMasking, self).__init__(max_time_mask_idx, False)
        else:
            super(FrequencyMasking, self).__init__(max_time_mask_idx, 1, False)

    def forward(self, specgram: Tensor) -> Tensor:
        mask_value = torch.rand(1)[0] * self.max_mask_value
        return super().forward(specgram, mask_value)


def _init_process(cfg, process):
    if process == 'logmel':
        return MelSpectrogram(cfg.sample_rate, cfg.n_fft, cfg.win_length, cfg.hop_length, cfg.f_min, cfg.f_max, pad=0,
                              n_mels=cfg.n_mels)
    elif process == 'delta':
        return ComputeDeltas(cfg.delta)
    elif process == 'time_mask':
        return TimeFreqMask(cfg.time_mask_len, cfg.mask_value, 'time')
    elif process == 'freq_mask':
        return TimeFreqMask(cfg.time_mask_len, cfg.mask_value, 'freq')
    elif process == 'time_stretch':
        return TimeStretch(hop_length=cfg.hop_length, n_freq=cfg.n_mels, fixed_rate=cfg.stretch_rate)
    elif process == 'normalize':
        return Normalize()
    else:
        raise NotImplementedError


class Normalize(torch.nn.Module):
    def forward(self, x: Tensor):
        return (x - x.mean()) / x.std()


class Transform(torch.nn.Module):
    # TODO GPU対応(Multiprocess対応, spawn)
    # TODO TimeStretchに対応するためにlogmelに複素数を返させる
    processes = {'logmel': MelSpectrogram, 'time_mask': TimeMasking, 'freq_mask': FrequencyMasking,
                 'normalize': Normalize}    # 'time_stretch': TimeStretch
    only_train_processes = ['time_mask', 'freq_mask', 'time_stretch']

    def __init__(self,
                 cfg: Dict,
                 phase: str) -> None:

        super(Transform, self).__init__()
        self.phase = phase
        self.cfg = cfg
        self.components = []
        self._init_components(cfg.transform_order)

    def _init_components(self, process_order):
        for process in process_order:
            if self.phase != 'train' and process in self.only_train_processes:
                continue

            self.components.append(
                _init_process(self.cfg, process)
            )

    def forward(self, x: Tensor):
        for component in self.components:
            x = component(x)
        x = x.unsqueeze(dim=0)
        return x
