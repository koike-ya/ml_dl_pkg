import torch

seed = 0
torch.manual_seed(seed)

torch.cuda.manual_seed_all(seed)
import random
random.seed(seed)
import torch.nn as nn


def type_int_list(args):
    return list(map(int, args.split(',')))


def nn_args(parser):
    nn_parser = parser.add_argument_group("CNN model arguments")

    # nn params
    nn_parser.add_argument('--nn-hidden-nodes', default='256,1028,200', type=type_int_list)

    return parser


from dataclasses import dataclass, field
from typing import List
from ml.utils.nn_config import NNModelConfig


@dataclass
class NNConfig(NNModelConfig):
    # TODO remove "nn_"
    nn_hidden_nodes: List[int] = field(default_factory=lambda: [256, 1028, 200])        # Early stopping with validation data


class NN(nn.Module):
    def __init__(self, hidden_nodes, in_features, n_classes=2):
        super(NN, self).__init__()

        self.fc = nn.Sequential()
        for i, n_nodes in enumerate(hidden_nodes + [n_classes]):
            in_features = in_features if i == 0 else hidden_nodes[i-1]
            self.fc.add_module(str(i), nn.Sequential(nn.Linear(in_features, n_nodes),
                                                     nn.ReLU(inplace=True),
                                                     nn.Dropout()))
        if n_classes >= 2:
            self.fc.add_module('softmax', nn.Softmax(dim=-1))

    def forward(self, x):
        return self.fc(x.float())


def construct_nn(cfg, use_as_extractor=False):
    return NN(cfg['nn_hidden_nodes'], cfg['input_size'], n_classes=len(cfg['class_names']))


