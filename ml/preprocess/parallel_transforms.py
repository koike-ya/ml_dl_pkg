from typing import List, Dict, Any
from dataclasses import dataclass, field

import torch
from torch import Tensor
from torchaudio.transforms import MelSpectrogram, TimeMasking

from ml.preprocess.transforms import Transform, TransConfig


def _init_process(cfg, process):
    if process == 'logmel':
        return MelSpectrogram(cfg.sample_rate, cfg.n_fft, cfg.win_length, cfg.hop_length, cfg.f_min, cfg.f_max, pad=0,
                              n_mels=cfg.n_mels)
    elif process == 'time_mask':
        return TimeMasking(cfg.time_mask_len)
    elif process == 'normalize':
        return Normalize()
    else:
        raise NotImplementedError


class Normalize(torch.nn.Module):
    def forward(self, x: Tensor):
        return (x - x.mean()) / x.std()


class ParallelTransform(torch.nn.Module):
    # TODO GPU対応(Multiprocess対応, spawn)
    def __init__(self,
                 cfg_list: List[TransConfig],
                 phase: str,
                 process_orders: List[List[str]]) -> None:

        super(ParallelTransform, self).__init__()
        self.cfg_list = cfg_list
        self.transforms = [Transform(cfg, phase, process) for cfg, process in zip(cfg_list, process_orders)]
        self.phase = phase

    def forward(self, x: Tensor):
        features = []
        for transform in self.transforms:
            features.append(transform(x))

        x = torch.cat(features, dim=0)
        return x
