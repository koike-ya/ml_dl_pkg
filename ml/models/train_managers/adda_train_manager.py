import argparse
import logging
import random
import sys
import time
from abc import ABCMeta
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

import numpy as np
import torch
from copy import deepcopy
from ml.models.base_model import model_args
from ml.models.ml_model_manager import MLModel
from ml.models.nn_model_manager import NNModel, supported_nn_models, supported_pretrained_models
from ml.models.train_manager import BaseTrainManager
from ml.models.nn_utils import set_requires_grad
from sklearn.metrics import confusion_matrix
from ml.utils.logger import TensorBoardLogger
from torch import nn
from tqdm import tqdm
from typing import Tuple, List, Union
from ml.utils.utils import Metrics

supported_ml_models = ['xgboost', 'knn', 'catboost', 'sgdc', 'lightgbm', 'svm']
supported_models = supported_ml_models + supported_nn_models + list(supported_pretrained_models.keys())


class AddaTrainManager(BaseTrainManager):
    def __init__(self, class_labels, cfg, dataloaders, metrics):
        super(AddaTrainManager, self).__init__(class_labels, cfg, dataloaders, metrics)
        self.src_enc = None
        self.tgt_enc = None
        self.disc = self._init_discriminator(self.tgt_clf.in_features)

    def _init_discriminator(self, in_features):
        return nn.Sequential(
            nn.Linear(in_features, 400),
            nn.ReLU(),
            nn.Linear(400, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        ).to(self.device)

    def _predict(self, phase) -> Tuple[np.array, np.array]:
        batch_size = self.cfg['batch_size']

        self.check_keys_from_dict([phase], self.dataloaders)

        dtype_ = np.int if self.cfg['task_type'] == 'classify' else np.float
        # ラベルが入れられなかった部分を除くため、小さな負の数を初期値として格納
        pred_list = np.zeros((len(self.dataloaders[phase]) * batch_size, 1), dtype=dtype_) - 1000000
        label_list = np.zeros((len(self.dataloaders[phase]) * batch_size, 1), dtype=dtype_) - 1000000
        for i, (inputs, labels) in tqdm(enumerate(self.dataloaders[phase]), total=len(self.dataloaders[phase])):

            inputs, labels = inputs.to(self.device), labels.numpy().reshape(-1,)
            preds = self.model.predict(inputs)
            pred_list[i * batch_size:i * batch_size + preds.shape[0], 0] = preds.reshape(-1,)
            label_list[i * batch_size:i * batch_size + labels.shape[0], 0] = labels

        pred_list, label_list = pred_list[~(pred_list == -1000000)], label_list[~(label_list == -1000000)]

        if self.cfg['tta']:
            pred_list = pred_list.reshape(self.cfg['tta'], -1).mean(axis=0)
            label_list = label_list[:label_list.shape[0] // self.cfg['tta']]

        return pred_list, label_list

    def train(self, model=None, with_validate=True) -> Tuple[Metrics, np.array]:

        super().train(with_validate=with_validate)
        self.src_enc = self.model
        self.tgt_enc = deepcopy(self.model)

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

                self.tgt_optimizer.zero_grad()
                disc_loss.backward()
                self.tgt_optimizer.step()

        self.raw_source_model.model.features = self.tgt_fe
        return self.raw_source_model

    def test(self, return_metrics=False, load_best=True, phase='test') -> Union[Tuple[np.array, np.array, Metrics],
                                                                                Tuple[np.array, np.array]]:
        if load_best:
            self.model.load_model()

        pred_list, label_list = self._predict(phase=phase)

        for metric in self.metrics['test']:
            if metric.name == 'loss':
                if self.cfg['task_type'] == 'classify':
                    y_onehot = torch.zeros(label_list.shape[0], len(self.class_labels))
                    y_onehot = y_onehot.scatter_(1, torch.from_numpy(label_list).view(-1, 1).type(torch.LongTensor), 1)
                    pred_onehot = torch.zeros(pred_list.shape[0], len(self.class_labels))
                    pred_onehot = pred_onehot.scatter_(1, torch.from_numpy(pred_list).view(-1, 1).type(torch.LongTensor), 1)
                    loss_value = self.model.criterion(pred_onehot.to(self.device), y_onehot.to(self.device)).item()
                elif self.cfg['model_type'] in ['rnn', 'cnn', 'cnn_rnn']:
                    loss_value = self.model.criterion(torch.from_numpy(pred_list).to(self.device),
                                                      torch.from_numpy(label_list).to(self.device))
            else:
                loss_value = 10000000

            metric.update(loss_value=loss_value, preds=pred_list, labels=label_list)
            logger.info(f"{phase} {metric.name}: {metric.average_meter.value :.4f}")
            metric.average_meter.update_best()

        if self.cfg['task_type'] == 'classify':
            confusion_matrix_ = confusion_matrix(label_list, pred_list,
                                                 labels=list(range(len(self.class_labels))))
            logger.info(confusion_matrix_)

        if return_metrics:
            return pred_list, label_list, self.metrics
        return pred_list, label_list

    def infer(self, load_best=True, phase='infer') -> np.array:
        if load_best:
            self.model.load_model()

        pred_list, _ = self._predict(phase=phase)

        return pred_list

    def retrain(self):
        phase = 'retrain'
        self.model.load_model()

        for metric in self.metrics:
            metric.add_average_meter(phase_name=phase)
            metric.add_average_meter(phase_name=f'{phase}_test')

            start = time.time()

        for epoch in range(self.cfg['retrain_epochs']):
            for i, (inputs, labels) in enumerate(self.dataloaders[phase]):

                loss, predicts = self.model.fit(inputs.to(self.device), labels.to(self.device), 'train')

                # save loss and metrics in one batch
                for metric in self.metrics[phase]:
                    metric.update(loss, predicts, labels.numpy())

                if not self.cfg['silent']:
                    self._verbose(epoch, phase, i, elapsed=int(time.time() - start))

            if self.logger:
                self._record_log(phase, epoch)

            self._update_by_epoch(phase, epoch, self.cfg['learning_anneal'])

        # selfのmetricsのretrain_testが更新される
        self.test(return_metrics=True, load_best=False, phase='retrain_test')

        return self.metrics
