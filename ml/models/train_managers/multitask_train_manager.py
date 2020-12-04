import logging
import time
from typing import Tuple

import torch
import numpy as np
import pandas as pd
from ml.models.model_managers.multitask_nn_model_manager import MultitaskNNModelManager
from ml.models.nn_models.nn import NNConfig
from ml.models.nn_models.cnn_rnn import CNNRNNConfig
from ml.models.train_managers.nn_train_manager import NNTrainManager
from ml.utils.utils import Metrics
from omegaconf import OmegaConf
from tqdm import tqdm
from copy import deepcopy

logger = logging.getLogger(__name__)


class MultitaskTrainManager(NNTrainManager):
    def __init__(self, class_labels, cfg, dataloaders, metrics):
        super(MultitaskTrainManager, self).__init__(class_labels, cfg, dataloaders, metrics)
        self.n_labels_in_each_task = cfg.model.n_labels_in_each_task
        self.n_tasks = len(self.n_labels_in_each_task)
        self.metrics_list = [deepcopy(self.metrics) for i in range(self.n_tasks)]

    def _init_model_manager(self) -> MultitaskNNModelManager:
        self.cfg.model.input_size = list(list(self.dataloaders.values())[0].get_input_size())

        if OmegaConf.get_type(self.cfg.model) in [CNNRNNConfig]:
            self.cfg.model.image_size = list(list(self.dataloaders.values())[0].get_image_size())
            self.cfg.model.in_channels = list(self.dataloaders.values())[0].get_n_channels()

            return MultitaskNNModelManager(self.class_labels, self.cfg.model)

        else:
            raise NotImplementedError

    def _verbose(self, epoch, phase, losses, i, elapsed, data_len=None) -> None:
        if not data_len:
            data_len = len(self.dataloaders[phase])

        eta = int(elapsed / (i + 1) * (data_len - (i + 1)))
        progress = f'\r{phase} epoch: [{epoch + 1}][{i + 1}/{data_len}]\t eta:{eta}(s)\t'
        progress += '\t'.join([f'task{i_task + 1} loss: {losses[i_task]:.4f}' for i_task in range(self.n_tasks)])
        logger.debug(progress)

    def predict(self, phase) -> Tuple[np.array, np.array]:
        pred_list, label_list = np.array([]), np.array([])
        for i, (inputs, labels) in tqdm(enumerate(self.dataloaders[phase]), total=len(self.dataloaders[phase])):
            inputs, labels = inputs.to(self.device), labels.numpy().reshape(-1, )
            preds = self.model_manager.predict(inputs)
            if pred_list.size == 0:
                pred_list = preds
            elif pred_list.ndim == 1:
                pred_list = np.hstack((pred_list, preds))
            else:
                pred_list = np.vstack((pred_list, preds))
            label_list = np.hstack((label_list, labels))

        if self.cfg.tta:
            pred_list, label_list = self._average_tta(pred_list, label_list)

        return pred_list, label_list

    def train(self, model_manager=None, with_validate=True, only_validate=False) -> Tuple[Metrics, np.array]:
        if model_manager:
            self.model_manager = model_manager

        start = time.time()
        best_val_pred = np.array([])

        if with_validate:
            phases = ['train', 'val']
        else:
            phases = ['train']
        if only_validate:
            phases = ['val']

        for epoch in range(self.cfg.epochs):
            for phase in phases:
                pred_list, label_list, loss_list = np.array([]), np.array([]), np.array([])

                for i, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    losses, predicts = self.model_manager.fit(inputs.to(self.device), labels, phase)
                    labels = np.array([labels[i_task].cpu().numpy() for i_task in range(self.n_tasks)])

                    if pred_list.size == 0:
                        pred_list = predicts
                        label_list = labels
                        loss_list = np.array(losses)[:, None]
                    else:
                        pred_list = np.hstack((pred_list, predicts))
                        label_list = np.hstack((label_list, labels))
                        loss_list = np.hstack((loss_list, np.array(losses)[:, None]))

                    self._verbose(epoch, phase, losses, i, elapsed=int(time.time() - start))

                # save metrics in one batch
                loss = loss_list.sum(axis=1).mean()
                for i_task in range(self.n_tasks):
                    for metric in self.metrics_list[i_task][phase]:
                        metric.update(loss, pred_list[i_task], label_list[i_task])

                    self._epoch_verbose(epoch, self.metrics_list[i_task], phase)

                    if self.logger:
                        self._record_log(phase, epoch, self.metrics_list[i_task], suffix=f'task-{i_task + 1}')

                best_val_flag = self._update_by_epoch(phase, self.metrics_list[0], self.cfg.model.optim.learning_anneal, epoch)

                if best_val_flag:
                    best_val_pred = pred_list[0].copy()
                    if not self.cfg.model.return_prob:
                        logger.debug(f'Best prediction of validation info:\n{pd.Series(best_val_pred).describe()}')

        if self.logger:
            self.logger.close()

        if not with_validate:
            self.model_manager.save_model()

        return self.metrics_list[0], best_val_pred
