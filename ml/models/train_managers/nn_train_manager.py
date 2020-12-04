import logging
import time
from contextlib import contextmanager
from typing import Tuple

import numpy as np
import pandas as pd
from ml.models.model_managers.nn_model_manager import NNModelManager
from ml.models.nn_models.cnn import CNNConfig
from ml.models.nn_models.cnn_rnn import CNNRNNConfig
from ml.models.nn_models.nn import NNConfig
from ml.models.nn_models.panns_cnn14 import PANNsConfig
from ml.models.nn_models.pretrained_models import PretrainedConfig
from ml.models.nn_models.rnn import RNNConfig
from ml.models.train_managers.base_train_manager import BaseTrainManager
from ml.utils.utils import Metrics
from omegaconf import OmegaConf
from tqdm import tqdm

logger = logging.getLogger(__name__)
CNN_MODELS = [CNNConfig, CNNRNNConfig, PretrainedConfig, PANNsConfig]


@contextmanager
def simple_timer(label) -> None:
    start = time.time()
    yield
    end = time.time()
    logger.info('{}: {:.3f}'.format(label, end - start))


class NNTrainManager(BaseTrainManager):
    def __init__(self, class_labels, cfg, dataloaders, metrics):
        super().__init__(class_labels, cfg, dataloaders, metrics)

    def _init_model_manager(self) -> NNModelManager:
        self.cfg.model.input_size = list(list(self.dataloaders.values())[0].get_input_size())

        if OmegaConf.get_type(self.cfg.model) in [RNNConfig]:
            if self.cfg.model.batch_norm_size:
                self.cfg.model.batch_norm_size = list(self.dataloaders.values())[0].get_batch_norm_size()
            self.cfg.model.seq_len = list(self.dataloaders.values())[0].get_seq_len()
        if OmegaConf.get_type(self.cfg.model) in [NNConfig] + CNN_MODELS:
            self.cfg.model.image_size = list(list(self.dataloaders.values())[0].get_image_size())
            self.cfg.model.in_channels = list(self.dataloaders.values())[0].get_n_channels()

            return NNModelManager(self.class_labels, self.cfg.model)
        else:
            raise NotImplementedError

    def _verbose(self, epoch, phase, metrics, i, elapsed, data_len=None) -> None:
        if not data_len:
            data_len = len(self.dataloaders[phase])
        eta = int(elapsed / (i + 1) * (data_len - (i + 1)))
        progress = f'\r{phase} epoch: [{epoch + 1}][{i + 1}/{data_len}]\t eta:{eta}(s)\t'
        progress += '\t'.join([f'{m.name} {m.average_meter.value:.4f}' for m in metrics[phase] if m.name == 'loss'])
        logger.debug(progress)

    def _update_by_epoch(self, phase, metrics, learning_anneal, epoch) -> bool:
        best_val_flag = False

        for metric in metrics[phase]:
            best_flag = metric.average_meter.update_best()
            if metric.save_model and best_flag and phase == 'val':
                logger.info(f"Found better validated model, saving to {self.cfg.model.model_path}")
                self.model_manager.save_model()
                best_val_flag = True

            # reset epoch average meter
            metric.average_meter.reset()

        if phase == 'train' and epoch + 1 in self.cfg.snapshot:
            orig_model_path = self.cfg.model.model_path
            self.cfg.model.model_path = self.cfg.model.model_path.replace('.pth', f'_ep{epoch + 1}.pth')
            self.model_manager.save_model()
            self.cfg.model.model_path = orig_model_path

        # anneal lr
        if phase == 'train':
            self.model_manager.anneal_lr(learning_anneal)

        return best_val_flag

    def _epoch_verbose(self, epoch, epoch_metrics, phase):
        message = f'epoch {str(epoch + 1).ljust(2)}-> lr: {self.model_manager.get_lr():.6f}\t'
        message += f'{phase}: ['
        message += '\t'.join([f'{m.name}: {m.average_meter.average:.4f}' for m in epoch_metrics[phase]])
        message += ']\t'

        if phase == 'train':
            logger.info(message)
        else:
            logger.info(message)

    def _predict(self, phase) -> Tuple[np.array, np.array]:
        pred_list, label_list = np.array([]), np.array([])
        for i, (inputs, labels) in tqdm(enumerate(self.dataloaders[phase]), total=len(self.dataloaders[phase])):
            inputs, labels = inputs.to(self.device), labels.numpy().reshape(-1,)
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

    def snapshot_predict(self, phase):
        snap_pred_list = []
        for epoch in self.cfg.snapshot:
            model_path = self.cfg.model.model_path
            self.cfg.model.model_path = self.cfg.model.model_path.replace('.pth', f'_ep{epoch}.pth')
            self.model_manager.load_model()
            self.cfg.model.model_path = model_path
            pred_list, label_list = self._predict(phase=phase)
            snap_pred_list.append(pred_list)

        ensemble = np.array(snap_pred_list).mean(axis=0)

        if not self.cfg.model.return_prob:
            ensemble = ensemble.astype(int)

        return ensemble, label_list

    def predict(self, phase):
        if self.cfg.snapshot:
            print('snapshot started')
            return self.snapshot_predict(phase=phase)
        else:
            return self._predict(phase=phase)

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
                pred_list, label_list = np.array([]), np.array([])

                for i, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    loss, predicts = self.model_manager.fit(inputs.to(self.device), labels.to(self.device), phase)

                    if labels.dim() == 2:     # If softlabel
                        labels = labels.argmax(dim=1)

                    if pred_list.size == 0:
                        pred_list = predicts
                    elif pred_list.ndim == 1:
                        pred_list = np.hstack((pred_list, predicts))
                    else:
                        pred_list = np.vstack((pred_list, predicts))
                    label_list = np.hstack((label_list, labels))
                    
                    # save loss in one batch
                    self.metrics[phase][0].update(loss, predicts, labels.numpy())

                    self._verbose(epoch, phase, self.metrics, i, elapsed=int(time.time() - start))

                # save metrics in one batch
                [metric.update(0.0, pred_list, label_list) for metric in self.metrics[phase][1:]]

                self._epoch_verbose(epoch, self.metrics, phase)

                if self.logger:
                    self._record_log(phase, epoch, self.metrics)

                best_val_flag = self._update_by_epoch(phase, self.metrics, self.cfg.model.optim.learning_anneal, epoch)

                if best_val_flag:
                    best_val_pred = pred_list.copy()
                    if not self.cfg.model.return_prob:
                        logger.debug(f'Best prediction of validation info:\n{pd.Series(best_val_pred).describe()}')

        if self.logger:
            self.logger.close()

        if not with_validate:
            self.model_manager.save_model()

        return self.metrics, best_val_pred
