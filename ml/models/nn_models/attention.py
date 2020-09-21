from dataclasses import dataclass, field
from typing import List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from ml.models.nn_models.nn_utils import initialize_weights, Predictor


@dataclass
class AttnConfig:    # Attention layer arguments
    d_attn: int = 64
    n_heads: int = 1


class Attention(nn.Module):
    def __init__(self, h_dim, d_attn, n_heads):
        super(Attention, self).__init__()
        self.h_dim = h_dim
        self.d_attn = d_attn
        self.n_heads = n_heads
        self.attn = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(h_dim, d_attn, bias=False),
            nn.Tanh(),
            nn.Linear(d_attn, n_heads, bias=False),
        )
        self.softmax = nn.Softmax(dim=2)
        
    def calc_attention(self, x):
        b_size = x.size(0)
        seq_len = x.size(1)
        attn_ene = self.attn(x.reshape(b_size * seq_len, self.h_dim))  # (b, s, h) -> (b * s, h_dim) -> (b * s, n_heads)
        x = attn_ene.view(b_size, seq_len, self.n_heads).transpose(1, 2)  # (b * s, n_heads) -> (b, s, n_heads) -> (b, n_heads, s)
        x = self.softmax(x)
        return x

    def forward(self, x):
        x = x.transpose(1, 2)  # (b, h, s) -> (b, s, h)
        attns = self.calc_attention(x)  # (b, s, h) -> (b, n_heads, s)
        feats = torch.bmm(attns, x).view(x.size(0), -1)

        return feats, attns


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads
        print(k.size())
        exit()

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        output = self.out(concat)

        return output


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class AttentionClassifier(nn.Module):
    def __init__(self, n_classes, h_dim, d_attn=512, n_heads=8):
        super(AttentionClassifier, self).__init__()
        self.attn = Attention(h_dim, d_attn, n_heads)
        # self.attn = MultiheadAttention(h_dim, n_heads, dropout=0.0)
        self.predictor = Predictor(h_dim * n_heads, n_classes, n_fc=1, tagging=False)

    def extract_feature(self, x):
        # x, _ = self.attn(x)
        # x = x.transpose(0, 2).transpose(1, 2)  # batch x feature x seq -> # seq x batch x feature
        x, _ = self.attn(x)
        # x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        x = self.extract_feature(x)
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
