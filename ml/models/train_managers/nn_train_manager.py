import logging
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from ml.models.train_managers.base_train_manager import BaseTrainManager
from tqdm import tqdm
from typing import Tuple
from ml.utils.utils import Metrics


@contextmanager
def simple_timer(label) -> None:
    start = time.time()
    yield
    end = time.time()
    logger.info('{}: {:.3f}'.format(label, end - start))


class NNTrainManager(BaseTrainManager):
    def __init__(self, class_labels, cfg, dataloaders, metrics):
        super().__init__(class_labels, cfg, dataloaders, metrics)

    def _verbose(self, epoch, phase, i, elapsed, data_len=None) -> None:
        if not data_len:
            data_len = len(self.dataloaders[phase])
        eta = int(elapsed / (i + 1) * (data_len - (i + 1)))
        progress = f'\r{phase} epoch: [{epoch + 1}][{i + 1}/{data_len}]\t eta:{eta}(s)\t'
        progress += '\t'.join([f'{m.name} {m.average_meter.value:.4f}' for m in self.metrics[phase] if m.name == 'loss'])
        logger.debug(progress)

    def _update_by_epoch(self, phase, learning_anneal, epoch) -> bool:
        best_val_flag = False

        for metric in self.metrics[phase]:
            best_flag = metric.average_meter.update_best()
            if metric.save_model and best_flag and phase == 'val':
                logger.info(f"Found better validated model, saving to {self.cfg.model.model_path}")
                self.model_manager.save_model()
                best_val_flag = True

            # reset epoch average meter
            metric.average_meter.reset()

        if phase == 'train' and epoch + 1 in self.cfg['snapshot']:
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

        if self.cfg['tta']:
            pred_list, label_list = self._average_tta(pred_list, label_list)

        return pred_list, label_list

    def snapshot_predict(self, phase):
        snap_pred_list = []
        for epoch in self.cfg['snapshot']:
            model_path = self.cfg.model.model_path
            self.cfg.model.model_path = self.cfg.model.model_path.replace('.pth', f'_ep{epoch}.pth')
            self.model_manager.load_model()
            self.cfg.model.model_path = model_path
            pred_list, label_list = self._predict(phase=phase)
            snap_pred_list.append(pred_list)

        ensemble = np.array(snap_pred_list).mean(axis=0)

        print(ensemble)
        if not self.cfg.model.return_prob:
            ensemble = ensemble.astype(int)
        print(ensemble)
        return ensemble, label_list

    def predict(self, phase):
        if self.cfg['snapshot']:
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

        for epoch in range(self.cfg['epochs']):
            for phase in phases:
                pred_list, label_list = np.array([]), np.array([])

                for i, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    loss, predicts = self.model_manager.fit(inputs.to(self.device), labels.to(self.device), phase)
                    if pred_list.size == 0:
                        pred_list = predicts
                    elif pred_list.ndim == 1:
                        pred_list = np.hstack((pred_list, predicts))
                    else:
                        pred_list = np.vstack((pred_list, predicts))
                    label_list = np.hstack((label_list, labels))
                    
                    # save loss in one batch
                    self.metrics[phase][0].update(loss, predicts, labels.numpy())

                    self._verbose(epoch, phase, i, elapsed=int(time.time() - start))

                # save metrics in one batch
                [metric.update(0.0, pred_list, label_list) for metric in self.metrics[phase][1:]]

                self._epoch_verbose(epoch, self.metrics, phase)

                if self.logger:
                    self._record_log(phase, epoch)

                best_val_flag = self._update_by_epoch(phase, self.cfg.model.optim.learning_anneal, epoch)

                if best_val_flag:
                    best_val_pred = pred_list.copy()
                    if not self.cfg.model.return_prob:
                        logger.debug(f'Best prediction of validation info:\n{pd.Series(best_val_pred).describe()}')

        if self.logger:
            self.logger.close()

        if not with_validate:
            self.model_manager.save_model()

        return self.metrics, best_val_pred

    def retrain(self):
        phase = 'retrain'
        self.model_manager.load_model()

        for metric in self.metrics:
            metric.add_average_meter(phase_name=phase)
            metric.add_average_meter(phase_name=f'{phase}_test')

            start = time.time()

        for epoch in range(self.cfg['retrain_epochs']):
            for i, (inputs, labels) in enumerate(self.dataloaders[phase]):

                loss, predicts = self.model_manager.fit(inputs.to(self.device), labels.to(self.device), 'train')

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
