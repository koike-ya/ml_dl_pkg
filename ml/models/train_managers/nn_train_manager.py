import logging
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)

import numpy as np
from copy import deepcopy
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
        progress = f'\r{phase} epoch: [{epoch + 1}][{i + 1}/{data_len}]\t {elapsed}(s) eta:{eta}(s)\t'
        progress += '\t'.join([f'{metric.name} {metric.average_meter.value:.4f}' for metric in self.metrics[phase]])
        logger.debug(progress)

    def _update_by_epoch(self, phase, epoch, learning_anneal) -> bool:
        best_val_flag = False

        for metric in self.metrics[phase]:
            best_flag = metric.average_meter.update_best()
            if metric.save_model and best_flag and phase == 'val':
                logger.info(f"Found better validated model, saving to {self.cfg['model_path']}")
                self.model_manager.save_model()
                best_val_flag = True

            # reset epoch average meter
            metric.average_meter.reset()

        # anneal lr
        if phase == 'train':
            self.model_manager.anneal_lr(learning_anneal)

        return best_val_flag

    def _epoch_verbose(self, epoch, epoch_metrics, phases):
        message = f'epoch {str(epoch + 1).ljust(2)}-> lr: {self.model_manager.get_lr():.6f}\t'
        for phase in phases:
            message += f'{phase}: ['
            message += '\t'.join([f'{m.name}: {m.average_meter.average:.4f}' for m in epoch_metrics[phase]])
            message += ']\t'
        logger.info(message)

    def _predict(self, phase) -> Tuple[np.array, np.array]:
        batch_size = self.cfg['batch_size']

        self.check_keys_from_dict([phase], self.dataloaders)

        dtype_ = np.int if self.cfg['task_type'] == 'classify' else np.float
        # ラベルが入れられなかった部分を除くため、小さな負の数を初期値として格納
        pred_list = np.zeros((len(self.dataloaders[phase]) * batch_size, 1), dtype=dtype_) - 1000000
        label_list = np.zeros((len(self.dataloaders[phase]) * batch_size, 1), dtype=dtype_) - 1000000
        for i, (inputs, labels) in tqdm(enumerate(self.dataloaders[phase]), total=len(self.dataloaders[phase])):

            inputs, labels = inputs.to(self.device), labels.numpy().reshape(-1,)
            preds = self.model_manager.predict(inputs)
            pred_list[i * batch_size:i * batch_size + preds.shape[0], 0] = preds.reshape(-1,)
            label_list[i * batch_size:i * batch_size + labels.shape[0], 0] = labels

        pred_list, label_list = pred_list[~(pred_list == -1000000)], label_list[~(label_list == -1000000)]

        if self.cfg['tta']:
            pred_list = pred_list.reshape(self.cfg['tta'], -1).mean(axis=0)
            label_list = label_list[:label_list.shape[0] // self.cfg['tta']]

        return pred_list, label_list

    def train(self, model_manager=None, with_validate=True, only_validate=False) -> Tuple[Metrics, np.array]:
        if model_manager:
            self.model_manager = model_manager

        start = time.time()
        epoch_metrics = {}
        best_val_pred = np.array([])

        if with_validate:
            phases = ['train', 'val']
        else:
            phases = ['train']
        if only_validate:
            phases = ['val']

        self.check_keys_from_dict(phases, self.dataloaders)
        batch_size = self.cfg['batch_size']
        dtype_ = np.int if self.cfg['task_type'] == 'classify' else np.float

        for epoch in range(self.cfg['epochs']):
            for phase in phases:
                # ラベルが入れられなかった部分を除くため、小さな負の数を初期値として格納
                pred_list = np.zeros((len(self.dataloaders[phase]) * batch_size,),
                                     dtype=dtype_) - 1000000

                for i, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    loss, predicts = self.model_manager.fit(inputs.to(self.device), labels.to(self.device), phase)
                    pred_list[i * batch_size:i * batch_size + predicts.shape[0]] = predicts

                    # save loss and metrics in one batch
                    for metric in self.metrics[phase]:
                        metric.update(loss, predicts, labels.numpy())

                    self._verbose(epoch, phase, i, elapsed=int(time.time() - start))

                if self.logger:
                    self._record_log(phase, epoch)

                epoch_metrics[phase] = deepcopy(self.metrics[phase])

                best_val_flag = self._update_by_epoch(phase, epoch, self.cfg['learning_anneal'])
                if best_val_flag:
                    best_val_pred = pred_list[~(pred_list == -1000000)]

            self._epoch_verbose(epoch, epoch_metrics, phases)

        if self.logger:
            self.logger.close()

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
