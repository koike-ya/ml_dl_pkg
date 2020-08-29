from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from ml.models.nn_models.nn_utils import initialize_weights, Predictor


@dataclass
class AttnConfig:    # Attention layer arguments
    da: int = 64
    n_heads: int = 1


class Attention(nn.Module):
    def __init__(self, h_dim, da, n_heads):
        super(Attention, self).__init__()
        self.h_dim = h_dim
        self.da = da
        self.n_heads = n_heads
        self.attn = nn.Sequential(
            nn.Linear(h_dim, da),
            nn.Tanh(),
            nn.Linear(da, n_heads)
        )
        
    def calc_attention(self, x):
        b_size = x.size(0)
        attn_ene = self.attn(x.reshape(-1, self.h_dim))  # (b, s, h) -> (b * s, n_heads)
        return F.softmax(attn_ene.view(b_size, -1, self.n_heads), dim=1)  # (b*s, n_heads) -> (b, s, n_heads)

    def forward(self, x):
        x = x.transpose(1, 2)  # (b, h, s) -> (b, s, h)
        attns = self.calc_attention(x)  # (b, s, h) -> (b, s, n_heads)
        feats = torch.stack(
            [(x * attns[:, :, i_head].unsqueeze(dim=2)).sum(dim=1) for i_head in range(self.n_heads)],  # (b, s, h) -> (b, h)
        dim=2)  # (b, h, n_heads)
        return feats, attns


class AttentionClassifier(nn.Module):
    def __init__(self, n_classes, h_dim, da=512, n_heads=8):
        super(AttentionClassifier, self).__init__()
        self.attn = Attention(h_dim, da, n_heads)
        self.predictor = Predictor(h_dim * n_heads, n_classes)

    def extract_feature(self, x):
        x, _ = self.attn(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        x, _ = self.feature_extract(x)
        return self.predictor(x)


class Attention2d(nn.Module):
    def __init__(self, n_in, n_out, att_activation, cla_activation):
        super(Attention2d, self).__init__()

        self.att_activation = att_activation
        self.cla_activation = cla_activation

        self.att = nn.Conv2d(
            in_channels=n_in, out_channels=n_out, kernel_size=(
                1, 1), stride=(
                1, 1), padding=(
                0, 0), bias=True)

        self.cla = nn.Conv2d(
            in_channels=n_in, out_channels=n_out, kernel_size=(
                1, 1), stride=(
                1, 1), padding=(
                0, 0), bias=True)

        self.init_weights()

    def init_weights(self):
        initialize_weights(self.att)
        initialize_weights(self.cla)
        self.att.weight.data.fill_(0.)

    def activate(self, x, activation):

        if activation == 'linear':
            return x

        elif activation == 'relu':
            return F.relu(x)

        elif activation == 'sigmoid':
            return F.sigmoid(x)+0.1

        elif activation == 'log_softmax':
            return F.log_softmax(x, dim=1)

    def forward(self, x):
        """input: (samples_num, channel, time_steps, freq_bins)
        """
        att = self.att(x)
        att = self.activate(att, self.att_activation)

        cla = self.cla(x)
        cla = self.activate(cla, self.cla_activation)

        # (samples_num, channel, time_steps * freq_bins)
        att = att.view(att.size(0), att.size(1), att.size(2) * att.size(3))
        cla = cla.view(cla.size(0), cla.size(1), cla.size(2) * cla.size(3))

        epsilon = 0.1 # 1e-7
        att = torch.clamp(att, epsilon, 1. - epsilon)

        norm_att = att / torch.sum(att, dim=2)[:, :, None]
        x = torch.sum(norm_att * cla, dim=2)

        Return_heatmap = False
        if Return_heatmap:
            return x, norm_att
        else:
            return x
