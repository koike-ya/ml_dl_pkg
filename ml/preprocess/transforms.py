from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import pandas as pd

import torch
from torch import Tensor
from torchaudio.transforms import MelSpectrogram, TimeMasking, FrequencyMasking, ComputeDeltas, TimeStretch, \
                                  AmplitudeToDB, Spectrogram, MelScale
from torchvision.transforms import RandomErasing

from ml.preprocess.heartsound_transforms import RespScale, HSTransConfig, RandomFlip
from ml.utils.enums import TimeFrequencyFeature
from ml.preprocess.augment import TimeFreqMask, AugConfig, Trim, RandomAmpChange


@dataclass
class TransConfig(AugConfig, HSTransConfig):
    sample_rate: float = 500.0  # The sample rate for the data/model features
    transform_order: List[str] = field(default_factory=lambda: ['trim', 'logmel', 'normalize'])
    n_fft: int = 800  # Size of FFT
    win_length: int = n_fft // 2     # Window size for spectrogram in data points
    hop_length: int = n_fft // 4  # Window stride for spectrogram in data points
    n_mels: int = 64            # Number of mel filters banks
    transform: TimeFrequencyFeature = TimeFrequencyFeature.none
    f_min: float = 0.0  # High pass filter
    f_max: float = sample_rate / 2  # Low pass filter
    delta: int = 5  # Compute delta coefficients of a tensor, usually a spectrogram
    stretch_rate: float = 1.0  # Time Stretch speedup/slow down rate

    random_erase_scale: Tuple[float, float] = (0.02, 0.33)
    random_erase_ratio: Tuple[float, float] = (0.3, 3.3)
    random_erase_prob: float = 0.5
    random_erase_value: float = 0.0


def _init_process(cfg, process):
    if process == 'trim':
        return Trim(cfg.sample_rate, cfg.trim_sec, False)
    elif process == 'random_trim':
        return Trim(cfg.sample_rate, cfg.trim_sec, True)
    elif process == 'random_amp_change':
        return RandomAmpChange(cfg.amp_change_prob, cfg.amp_change_scale)
    elif process == 'spectrogram':
        return Spectrogram(cfg.n_fft, cfg.win_length, cfg.hop_length)
    elif process == 'mel_scale':
        return MelScale(cfg.n_mels, cfg.sample_rate, cfg.f_min, cfg.f_max, cfg.n_fft // 2 + 1)
    elif process == 'delta':
        return ComputeDeltas(cfg.delta)
    elif process == 'time_mask':
        return TimeFreqMask(cfg.spec_aug_prob, cfg.time_mask_len, cfg.mask_value, cfg.n_time_mask, 'time')
    elif process == 'freq_mask':
        return TimeFreqMask(cfg.spec_aug_prob, cfg.freq_mask_len, cfg.mask_value, cfg.n_freq_mask, 'freq')
    elif process == 'time_stretch':
        return TimeStretch(hop_length=cfg.hop_length, n_freq=cfg.n_mels, fixed_rate=cfg.stretch_rate)
    elif process == 'normalize':
        return Normalize()
    elif process == 'power_to_db':
        return AmplitudeToDB()
    elif process == 'random_erase':
        return RandomErasing(cfg.random_erase_prob, cfg.random_erase_scale, cfg.random_erase_ratio,
                             cfg.random_erase_value)
    elif process == 'resp_scale':
        return RespScale(cfg.resp_p, cfg.sample_rate, cfg.resp_period, cfg.resp_amp_scale)
    elif process == 'random_flip':
        return RandomFlip(cfg.flip_p)
    else:
        raise NotImplementedError


class Normalize(torch.nn.Module):
    def forward(self, x: Tensor):
        return (x - x.mean()) / x.std()


class Transform(torch.nn.Module):
    # TODO GPU対応(Multiprocess対応, spawn)
    # TODO TimeStretchに対応するためにlogmelに複素数を返させる
    processes = ['trim', 'random_trim', 'random_amp_change', 'resp_scale', 'random_flip',
                 'spectrogram', 'mel_scale', 'time_mask', 'freq_mask',
                 'power_to_db', 'random_erase', 'normalize']    # 'time_stretch': TimeStretch
    only_train_processes = ['random_amp_change', 'resp_scale', 'time_mask', 'freq_mask', 'time_stretch', 'random_erase']

    def __init__(self,
                 cfg: Dict,
                 phase: str) -> None:

        super(Transform, self).__init__()
        self.phase = phase
        self.cfg = cfg
        self.components = []
        assert pd.Series(cfg.transform_order).isin(Transform.processes).all()
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
            if x.ndim == 2:
                x = x.unsqueeze(dim=0)
        return x
