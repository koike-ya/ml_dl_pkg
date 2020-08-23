from dataclasses import dataclass, field
from typing import List

import torch.nn as nn


@dataclass
class MultitaskConfig:
    n_tasks: int = 2
    n_labels_in_each_task: List[int] = field(default_factory=lambda: [2, 2])


class MultitaskPredictor(nn.Module):
    def __init__(self, in_features: int, n_tasks: int, n_labels_in_each_task: List[int]):
        super(MultitaskPredictor, self).__init__()

        self.predictors = [nn.Linear(in_features, n_labels_in_each_task[i]) for i in range(n_tasks)]
        for i in range(n_tasks):
            if n_labels_in_each_task[i] >= 2:
                self.predictors[i] = nn.Sequential(
                    self.predictors[i],
                    nn.Softmax(dim=-1)
                )

    def forward(self, x):
        return (self.predictors[i](x) for i in range(len(self.predictors)))
