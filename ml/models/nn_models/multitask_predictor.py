from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn


@dataclass
class MultitaskConfig:
    n_labels_in_each_task: List[int] = field(default_factory=lambda: [2, 2])


class MultitaskPredictor(nn.Module):
    def __init__(self, in_features: int, n_labels_in_each_task: List[int], device: torch.device):
        super(MultitaskPredictor, self).__init__()

        n_tasks = len(n_labels_in_each_task)
        self.predictors = [nn.Linear(in_features, n_labels_in_each_task[i]) for i in range(n_tasks)]
        for i in range(n_tasks):
            if n_labels_in_each_task[i] >= 2:
                self.predictors[i] = nn.Sequential(
                    self.predictors[i],
                    nn.Softmax(dim=-1)
                ).to(device)

    def forward(self, x):
        preds = []
        for i in range(len(self.predictors)):
            device = self.predictors[i][0].weight.device
            preds.append(self.predictors[i](x.to(device)))
        return preds
