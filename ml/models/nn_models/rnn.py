from __future__ import print_function, division

from collections import OrderedDict

import torch
import torch.nn as nn

from ml.models.nn_models.nn_utils import initialize_weights

supported_rnns = {
    'lstm': nn.LSTM,
    'rnn': nn.RNN,
    'gru': nn.GRU
}


from dataclasses import dataclass
from ml.utils.enums import RNNType
from ml.utils.nn_config import NNModelConfig
from ml.models.nn_models.nn_utils import Predictor


@dataclass
class RNNConfig(NNModelConfig):    # RNN model arguments
    # TODO remove "rnn_"
    rnn_type: RNNType = RNNType.gru     # Type of the RNN. rnn|gru|lstm|deepspeech are supported
    rnn_hidden_size: int = 100      # Hidden size of RNNs
    rnn_n_layers: int = 1  # Number of RNN layers
    dropout: float = 0.5  # Dropout after each rnn layer
    max_norm: int = 400     # Norm cutoff to prevent explosion of gradients
    bidirectional: bool = True      # Turn off bi-directional RNNs, introduces lookahead convolution
    # TODO change to bn
    batch_norm_size: int = 0   # Batch normalization or not
    seq_len: int = 0  # Length of sequence


def construct_rnn(cfg, output_size):
    """

    :param cfg: {
        'rnn_type': 'deepspeech' or lstm or gru,
        'input_size': input feature size of data
        'n_layers': Number of layers in rnn
        'seq_len': Length of time dimension
        'hidden_size': Number of hidden size in rnn
        'is_bidirectional': True or False
        'is_inference_softmax': True or False
    }
    :return:
    """
    if len(cfg.input_size) == 2:
        cfg.input_size = cfg.input_size[0]

    return RNNClassifier(cfg.input_size, out_time_feature=cfg.seq_len,
                         rnn_type=supported_rnns[cfg.rnn_type.value], n_classes=output_size,
                         rnn_hidden_size=cfg.rnn_hidden_size, n_layers=cfg.rnn_n_layers,
                         bidirectional=cfg.bidirectional, dropout=cfg.dropout)


class RNNClassifier(nn.Module):
    def __init__(self, input_size, out_time_feature, n_classes, rnn_type=nn.LSTM, rnn_hidden_size=768, n_layers=5,
                 bidirectional=True, dropout=0.3):
        super(RNNClassifier, self).__init__()

        self.rnn = initialize_weights(
            rnn_type(input_size=input_size, hidden_size=rnn_hidden_size, bidirectional=bidirectional, bias=True,
                     dropout=dropout, num_layers=n_layers))

        rnn_hidden_size = rnn_hidden_size * 2 if bidirectional else rnn_hidden_size
        self.predictor = Predictor(in_features=rnn_hidden_size * out_time_feature, n_classes=n_classes)

    def extract_feature(self, x):
        x = x.transpose(0, 2).transpose(1, 2)  # batch x feature x seq -> # seq x batch x feature
        x, _ = self.rnn(x)
        x = x.transpose(0, 1).transpose(1, 2)  # seq x batch x feature -> batch x seq x feature -> batch x feature x seq

        return x

    def predict(self, x):
        x = x.reshape(x.size(0), -1)
        return self.predictor(x)

    def forward(self, x):
        x = self.extract_feature(x)
        x = self.predict(x)

        return x
