from dataclasses import dataclass

import torch
from torch import nn

from ml.models.nn_models.cnn import CNNConfig
from ml.models.nn_models.nn_utils import get_param_size
from ml.models.nn_models.rnn import RNNClassifier, supported_rnns, RNNConfig


@dataclass
class CNNRNNConfig(CNNConfig, RNNConfig):
    pass


class DeepSpeech(RNNClassifier):
    def __init__(self, conv, input_size, out_time_feature, rnn_type=nn.LSTM,
                 rnn_hidden_size=768, n_layers=5, bidirectional=True, output_size=2):
        super(DeepSpeech, self).__init__(input_size=input_size, out_time_feature=out_time_feature,
                                         rnn_type=rnn_type, rnn_hidden_size=rnn_hidden_size, n_layers=n_layers,
                                         bidirectional=bidirectional, output_size=output_size, batch_norm_size=input_size)

        self.hidden_size = rnn_hidden_size
        self.hidden_layers = n_layers
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional

        self.conv = conv
        print(f'Number of parameters\tconv: {get_param_size(self.conv)}\trnn: {get_param_size(super())}')

    def forward(self, x):
        if len(x.size()) <= 2:
            x = torch.unsqueeze(x, dim=1)

        x = self.conv(x.to(torch.float))    # batch x channel x time x freq

        if len(x.size()) == 4:      # batch x channel x time_feature x freq_feature
            # Collapse feature dimension   batch x feature x time
            x = x.transpose(2, 3)
            sizes = x.size()
            x = x.reshape(sizes[0], sizes[1] * sizes[2], sizes[3])

        x = super().forward(x)
        return x


def construct_cnn_rnn(cfg, construct_cnn_func, output_size, device):
    conv, conv_out_ftrs = construct_cnn_func(cfg, use_as_extractor=True)
    input_size = conv_out_ftrs['n_channels'] * conv_out_ftrs['width']
    return DeepSpeech(conv.to(device), input_size, out_time_feature=conv_out_ftrs['height'],
                      rnn_type=supported_rnns[cfg.rnn_type.value], rnn_hidden_size=cfg.rnn_hidden_size,
                      n_layers=cfg.rnn_n_layers, bidirectional=cfg.bidirectional, output_size=output_size)
