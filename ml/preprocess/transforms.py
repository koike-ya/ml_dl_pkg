from dataclasses import dataclass
from typing import List, Dict

import torch
from torch import Tensor
from torchaudio.transforms import MelSpectrogram, TimeMasking, ComputeDeltas

from ml.utils.enums import TimeFrequencyFeature


@dataclass
class TransConfig:
    sample_rate: float = 500.0  # The sample rate for the data/model features
    n_fft: int = 800  # Size of FFT
    win_length: int = n_fft    # Window size for spectrogram in data points
    hop_length: int = n_fft // 2  # Window stride for spectrogram in data points
    n_mels: int = 64            # Number of mel filters banks
    transform: TimeFrequencyFeature = TimeFrequencyFeature.none
    f_min: float = 0.0  # High pass filter
    f_max: float = sample_rate / 2  # Low pass filter
    time_mask_len: int = 10  # maximum possible length of the mask. Indices uniformly sampled from [0, time_mask_param)
    delta: int = 5  # Compute delta coefficients of a tensor, usually a spectrogram


def _init_process(cfg, process):
    if process == 'logmel':
        return MelSpectrogram(cfg.sample_rate, cfg.n_fft, cfg.win_length, cfg.hop_length, cfg.f_min, cfg.f_max, pad=0,
                              n_mels=cfg.n_mels)
    elif process == 'delta':
        return ComputeDeltas(cfg.delta)
    elif process == 'time_mask':
        return TimeMasking(cfg.time_mask_len)
    elif process == 'normalize':
        return Normalize()
    else:
        raise NotImplementedError


class Normalize(torch.nn.Module):
    def forward(self, x: Tensor):
        return (x - x.mean()) / x.std()


class Transform(torch.nn.Module):
    # TODO GPU対応(Multiprocess対応, spawn)
    processes = {'logmel': MelSpectrogram, 'time_mask': TimeMasking, 'normalize': Normalize}
    only_train_processes = ['time_mask']

    def __init__(self,
                 cfg: Dict,
                 phase: str,
                 process_order: List[str]) -> None:

        super(Transform, self).__init__()
        self.phase = phase
        self.cfg = cfg
        self.components = []
        self._init_components(process_order)

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
