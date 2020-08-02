from dataclasses import dataclass
from typing import List, Dict

import torch
from torch import Tensor
from torchaudio.transforms import MelSpectrogram, TimeMasking

from ml.utils.enums import TimeFrequencyFeature

processes = {'logmel': MelSpectrogram, 'time_mask': TimeMasking}
only_train_processes = ['time_mask']


@dataclass
class TransConfig:
    sample_rate: float = 500.0  # The sample rate for the data/model features
    n_fft: int = 400  # Size of FFT
    win_length: int = n_fft    # Window size for spectrogram in seconds
    hop_length: int = n_fft // 2 + 1  # Window stride for spectrogram in seconds
    n_mels: int = 64            # Number of mel filters banks
    transform: TimeFrequencyFeature = TimeFrequencyFeature.none
    f_min: float = 0.0  # High pass filter
    f_max: float = sample_rate / 2  # Low pass filter
    time_mask_len: int = 1000  # maximum possible length of the mask. Indices uniformly sampled from [0, time_mask_param)


def _init_process(cfg, process):
    if process == 'logmel':
        return MelSpectrogram(cfg.sample_rate, cfg.n_fft, cfg.win_length, cfg.hop_length, cfg.f_min, cfg.f_max, pad=0,
                              n_mels=cfg.n_mels)
    elif process == 'time_mask':
        return TimeMasking(cfg.time_mask_len)
    else:
        raise NotImplementedError


class Transform(torch.nn.Module):
    # TODO GPU対応
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
            if self.phase != 'train' and process in only_train_processes:
                continue

            self.components.append(
                _init_process(self.cfg, process)
            )

    def forward(self, x: Tensor):
        for component in self.components:
            x = component(torch.tensor(x))

        x = x.unsqueeze(dim=0)
        return x
