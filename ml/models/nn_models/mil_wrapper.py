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


class EmbeddingAggMil(nn.Module):
    def __init__(self, model, agg_func=MilAggType.mean):
        super(EmbeddingAggMil, self).__init__()
        self.model = model
        self.agg_func = agg_func

    def forward(self, bag):
        # TODO extend more than 2 classes
        bag = bag.squeeze(0)
        bag_features = self.model.extract_feature(bag)

        if self.agg_func == MilAggType.mean:
            pred = self.model.classify(bag_features.mean(dim=0))
        elif self.agg_func == MilAggType.max:
            pred = self.model.classify(torch.max(bag_features, dim=0)[0])
        else:
            raise NotImplementedError

        prob = torch.sigmoid(pred)
        prob = prob / prob.sum()

        return prob.unsqueeze(dim=0)


class AttentionMil(nn.Module):
    def __init__(self, model):
        super(AttentionMil, self).__init__()
        self.model = model
        self.L = 2048
        self.D = 128
        self.K = 2
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, 2),
            nn.Sigmoid()
        )

    def forward(self, bag):
        # TODO extend more than 2 classes
        bag = bag.squeeze(0)
        H = self.model.extract_feature(bag)

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL
        M = M.view(1, -1)
        prob = self.classifier(M)
        prob = prob / prob.sum()

        return prob


def construct_mil(model, cfg):
    if cfg.mil_type == MilType.instance:
        return InstanceAggMil(model, cfg.mil_agg_func)
    elif cfg.mil_type == MilType.embedding:
        return EmbeddingAggMil(model, cfg.mil_agg_func)
    elif cfg.mil_type == MilType.attention:
        return AttentionMil(model)
    else:
        raise NotImplementedError
