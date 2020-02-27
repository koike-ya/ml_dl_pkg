"""
in: 学習済みモデルとちゃんとtarget用のアノテーションされたデータ
out: よりdomain不変な特徴量を作成できるようになった学習済みモデル
"""

"""
Implements ADDA:
Adversarial Discriminative Domain Adaptation, Tzeng et al. (2017)
"""

from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm, trange

from ml.models.model_managers.nn_model_manager import NNModelManager


DOMAIN_ADAPTATION_LABELS = ['source', 'target']


def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


class AddaModelManager(NNModelManager):
    def __init__(self, class_labels, cfg):
        super().__init__(class_labels, cfg)
        self.src = None
        self.tgt = None
        # self.tgt_fe, self.tgt_clf = self._init_tgt_model()
        self.tgt_optimizer = deepcopy(self.optimizer)
        self.disc = self._init_discriminator(in_features=2048)
        self.discriminator_optim = torch.optim.Adam(self.disc.parameters())
        self.disc_criterion = nn.BCEWithLogitsLoss()
    #
    # def _init_src_model(self):
    #     _ = list(self.model.children())
    #     feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
    #     src_model.model.eval()
    #     set_requires_grad(src_model.model, requires_grad=False)
    #     return src_model.model.features, src_model.model
    #
    # def _init_tgt_model(self):
    #     target_feature_extractor = deepcopy(self.src_model).features
    #     target_classifier = deepcopy(self.src_model).classifier
    #     target_classifier.eval()
    #     return target_feature_extractor, target_classifier

    def _init_discriminator(self, in_features):
        return nn.Sequential(
            nn.Linear(in_features, 2048),
            nn.ReLU(),
            nn.Linear(2048, 200),
            nn.ReLU(),
            nn.Linear(200, 1)
        ).to(self.device)

    def fit_discriminator(self, src_x, tgt_x):
        if not self.src:
            self.src = self.model
            self.tgt = deepcopy(self.model)

        set_requires_grad(self.tgt.feature_extractor, requires_grad=False)
        set_requires_grad(self.disc, requires_grad=True)

        src_features = self.src.feature_extract(src_x).view(src_x.shape[0], -1)
        tgt_features = self.tgt.feature_extract(tgt_x).view(tgt_x.shape[0], -1)

        discriminator_x = torch.cat([src_features, tgt_features])
        discriminator_y = torch.cat([torch.zeros(src_x.shape[0], device=self.device),
                                     torch.ones(tgt_x.shape[0], device=self.device)])

        disc_preds = self.disc(discriminator_x).squeeze()
        disc_loss = self.disc_criterion(disc_preds, discriminator_y)

        self.discriminator_optim.zero_grad()
        disc_loss.backward()
        self.discriminator_optim.step()

        return disc_loss.item()

    def fit_classifier(self, tgt_x):
        set_requires_grad(self.tgt.feature_extractor, requires_grad=True)
        set_requires_grad(self.disc, requires_grad=False)

        tgt_features = self.tgt.feature_extract(tgt_x).view(tgt_x.shape[0], -1)

        # flipped labels
        discriminator_y = torch.zeros(tgt_x.shape[0], device=self.device)

        preds = self.disc(tgt_features).squeeze()
        disc_loss = self.disc_criterion(preds, discriminator_y)

        self.tgt_optimizer.zero_grad()
        disc_loss.backward()
        self.tgt_optimizer.step()

        return disc_loss.item()
