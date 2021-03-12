from dataclasses import dataclass
from pathlib import Path
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.utils.enums import MilType, MilAggType


class InstanceAggMil(nn.Module):
    def __init__(self, model, agg_func=MilAggType.mean):
        super(InstanceAggMil, self).__init__()
        self.model = model
        self.agg_func = agg_func

    def forward(self, bag):
        # TODO extend more than 2 classes
        bag = bag.squeeze(0)
        prob_instances = self.model(bag)

        prob_instances = torch.sigmoid(prob_instances)
        prob_instances = (prob_instances.T / prob_instances.sum(dim=1)).T

        if self.agg_func == MilAggType.mean:
            pred = prob_instances.mean(dim=0).unsqueeze(0)
            pred = (pred.T / pred.sum(dim=1)).T
        elif self.agg_func == MilAggType.max:
            pred = prob_instances[prob_instances[:, 1].argmax()].unsqueeze(0)
        else:
            raise NotImplementedError

        return pred


def construct_mil(model, cfg):
    if cfg.mil_type == MilType.instance:
        return InstanceAggMil(model, cfg.mil_agg_func)
    else:
        raise NotImplementedError
