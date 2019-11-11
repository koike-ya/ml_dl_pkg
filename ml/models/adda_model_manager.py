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

from ml.models.model_manager import BaseModelManager


DOMAIN_ADAPTATION_LABELS = ['source', 'target']


def loop_iterable(iterable):
    while True:
        yield from iterable


def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


class AddaModelManager(BaseModelManager):
    def __init__(self, source_model, cfg, dataloaders, metrics):
        super().__init__(DOMAIN_ADAPTATION_LABELS, cfg, dataloaders, metrics)
        self.src_fe, self.src_model = self._init_src_model(source_model)
        self.tgt_fe, self.tgt_clf = self._init_tgt_model()
        self.disc = self._init_discriminator(self.tgt_clf.in_features)

    def _init_src_model(self, src_model):
        src_model.model.eval()
        set_requires_grad(src_model.model, requires_grad=False)
        return src_model.model.features, src_model.model

    def _init_tgt_model(self):
        target_feature_extractor, target_classifier = deepcopy(self.src_model), deepcopy(self.src_model)
        target_feature_extractor.model = target_feature_extractor.model.features
        target_classifier.model = target_classifier.model.classifier
        target_classifier.eval()
        return target_feature_extractor, target_classifier

    def _init_discriminator(self, in_features):
        return nn.Sequential(
            nn.Linear(in_features, 400),
            nn.ReLU(),
            nn.Linear(400, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        ).to(self.device)

    def train(self):
        discriminator_optim = torch.optim.Adam(self.disc.parameters())
        disc_criterion = nn.BCEWithLogitsLoss()

        batch_iterator = zip(loop_iterable(self.dataloaders['source']), loop_iterable(self.dataloaders['target']))

        for _ in trange(self.cfg['iterations'], leave=False):
            # Train discriminator
            set_requires_grad(self.tgt_fe, requires_grad=False)
            set_requires_grad(self.disc, requires_grad=True)
            for _ in range(self.cfg['k_disc']):
                (source_x, _), (target_x, _) = next(batch_iterator)
                source_x, target_x = source_x.to(self.device), target_x.to(self.device)

                source_features = self.src_fe(source_x).view(source_x.shape[0], -1)
                target_features = self.tgt_fe(target_x).view(target_x.shape[0], -1)

                discriminator_x = torch.cat([source_features, target_features])
                discriminator_y = torch.cat([torch.ones(source_x.shape[0], device=self.device),
                                             torch.zeros(target_x.shape[0], device=self.device)])

                disc_preds = self.disc(discriminator_x).squeeze()
                disc_loss = disc_criterion(disc_preds, discriminator_y)

                discriminator_optim.zero_grad()
                disc_loss.backward()
                discriminator_optim.step()

                disc_loss += disc_loss.item()

            # Train classifier
            set_requires_grad(self.tgt_fe, requires_grad=True)
            set_requires_grad(self.disc, requires_grad=False)

            for _ in range(self.cfg['k_clf']):
                _, (target_x, _) = next(batch_iterator)
                target_x = target_x.to(self.device)
                target_features = self.tgt_fe(target_x).view(target_x.shape[0], -1)

                # flipped labels
                discriminator_y = torch.ones(target_x.shape[0], device=self.device)

                preds = self.disc(target_features).squeeze()
                disc_loss = disc_criterion(preds, discriminator_y)

                self.tgt_clf.optimizer.zero_grad()
                disc_loss.backward()
                self.tgt_clf.optimizer.step()

        self.src_model.model.features = self.tgt_fe.model
        return self.src_model