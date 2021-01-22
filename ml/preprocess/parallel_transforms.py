from typing import List, Dict, Any
from dataclasses import dataclass, field

import torch
from torch import Tensor
from torchaudio.transforms import MelSpectrogram, TimeMasking

from ml.preprocess.transforms import Transform, TransConfig


class ParallelTransform(torch.nn.Module):
    # TODO GPU対応(Multiprocess対応, spawn)
    def __init__(self,
                 cfg_list: List[TransConfig],
                 phase: str) -> None:

        super(ParallelTransform, self).__init__()
        self.cfg_list = cfg_list
        self.transforms = [Transform(cfg, phase) for cfg in cfg_list]
        self.phase = phase

    def forward(self, x: Tensor):
        features = []
        for transform in self.transforms:
            features.append(transform(x))

        if len(features):
            x = torch.cat(features, dim=0)

        return x
