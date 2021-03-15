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

    def forward(self, bags):
        # TODO extend more than 2 classes
        pred_list = []
        for bag in bags:
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
            pred_list.append(pred)

        pred_list = torch.cat(pred_list, dim=0)

        return pred_list


class EmbeddingAggMil(nn.Module):
    def __init__(self, model, agg_func=MilAggType.mean):
        super(EmbeddingAggMil, self).__init__()
        self.model = model
        self.agg_func = agg_func

    def forward(self, bags):
        # TODO extend more than 2 classes
        pred_list = []
        for bag in bags:
            bag_features = self.model.extract_feature(bag)

            if self.agg_func == MilAggType.mean:
                pred = self.model.classify(bag_features.mean(dim=0))
            elif self.agg_func == MilAggType.max:
                pred = self.model.classify(torch.max(bag_features, dim=0)[0])
            else:
                raise NotImplementedError
            prob = torch.sigmoid(pred.unsqueeze(0))
            prob = prob / prob.sum()

            pred_list.append(prob)

        pred_list = torch.cat(pred_list, dim=0)
        return pred_list


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

    def forward(self, bags):
        # TODO extend more than 2 classes
        pred_list = []
        for bag in bags:
            H = self.model.extract_feature(bag)

            A = self.attention(H)  # NxK
            A = torch.transpose(A, 1, 0)  # KxN
            A = F.softmax(A, dim=1)  # softmax over N

            M = torch.mm(A, H)  # KxL
            M = M.view(1, -1)
            prob = self.classifier(M)
            prob = prob / prob.sum()
            print(prob.size())
            pred_list.append(prob)

        pred_list = torch.cat(pred_list, dim=0)
        return pred_list


def construct_mil(model, cfg):
    if cfg.mil_type == MilType.instance:
        return InstanceAggMil(model, cfg.mil_agg_func)
    elif cfg.mil_type == MilType.embedding:
        return EmbeddingAggMil(model, cfg.mil_agg_func)
    elif cfg.mil_type == MilType.attention:
        return AttentionMil(model)
    else:
        raise NotImplementedError
