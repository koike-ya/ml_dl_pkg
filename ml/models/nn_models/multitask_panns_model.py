import logging

import torch
import torch.nn.functional as F
from torch import nn

from ml.models.nn_models.panns_cnn14 import Cnn14

logger = logging.getLogger(__name__)


class MultitaskPanns(Cnn14):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
                 fmax, classes_num, checkpoint_path):
        super(MultitaskPanns, self).__init__(sample_rate, window_size, hop_size, mel_bins, fmin,
                 fmax, classes_num, checkpoint_path)

        self.fc_valence = nn.Linear(2048, 2048, bias=True)
        self.fc_arousal = nn.Linear(2048, 2048, bias=True)
        self.classifier_valence = nn.Linear(2048, classes_num, bias=True)
        self.classifier_arousal = nn.Linear(2048, classes_num, bias=True)

    def forward(self, input, mixup_lambda=None):
        """
                Input: (batch_size, data_length)"""
        x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x_v = F.relu_(self.fc_valence(x))
        x_a = F.relu_(self.fc_arousal(x))

        pred_v = self.classifier_valence(x_v)
        pred_a = self.classifier_valence(x_a)

        return pred_v, pred_a


def construct_multitask_panns(cfg):
    sample_rate = cfg['sample_rate']
    window_size = cfg['window_size'] * sample_rate
    hop_size = cfg['window_stride'] * sample_rate
    mel_bins = cfg['n_mels']
    fmin = cfg['low_cutoff']
    fmax = cfg['high_cutoff']

    checkpoint_path = cfg['checkpoint_path']
    device = torch.device('cuda') if cfg['cuda'] and torch.cuda.is_available() else torch.device('cpu')

    model = MultitaskPanns(sample_rate=sample_rate, window_size=window_size, hop_size=hop_size, mel_bins=mel_bins,
                             fmin=fmin, fmax=fmax, classes_num=len(cfg['class_names']),
                             checkpoint_path=checkpoint_path).to(device)

    return model
