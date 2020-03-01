import logging

logger = logging.getLogger(__name__)

import numpy as np
from typing import Tuple
from ml.utils.utils import Metrics
from ml.models.train_managers.base_train_manager import BaseTrainManager


class MLTrainManager(BaseTrainManager):
    def __init__(self, class_labels, cfg, dataloaders, metrics):
        super().__init__(class_labels, cfg, dataloaders, metrics)
    
    def _predict(self, phase) -> Tuple[np.array, np.array]:
        # dtype_ = np.int if self.cfg['task_type'] == 'classify' else np.float
        # # ラベルが入れられなかった部分を除くため、小さな負の数を初期値として格納
        # pred_list = np.zeros((len(self.dataloaders[phase]) * batch_size, 1), dtype=dtype_) - 1000000
        # label_list = np.zeros((len(self.dataloaders[phase]) * batch_size, 1), dtype=dtype_) - 1000000
        
        inputs = [data[0] for data in self.dataloaders[phase]]
        label_list = [data[1] for data in self.dataloaders[phase]]

        pred_list = self.model_manager.predict(inputs)
        
        if self.cfg['tta']:
            pred_list = pred_list.reshape(self.cfg['tta'], -1).mean(axis=0)
            label_list = label_list[:label_list.shape[0] // self.cfg['tta']]

        return pred_list, label_list

    def train(self, model_manager=None, with_validate=True, only_validate=False) -> Tuple[
        Metrics, np.array]:
        if model_manager:
            self.model_manager = model_manager
            
        self.check_keys_from_dict(phases, self.dataloaders)

        if with_validate:
            phases = ['train', 'val']
        else:
            phases = ['train']
        if only_validate:
            phases = ['val']

        inputs, labels = {}, {}
        for phase in phases:
            inputs[phase] = np.vstack(tuple([data[0].numpy() for data in self.dataloaders[phase]]))
            labels[phase] = np.hstack(tuple([data[1].numpy() for data in self.dataloaders[phase]]))

        if 'train' in phases:
            loss = self.model_manager.fit(inputs['train'], labels['train'])
            predicts = self.model_manager.predict(inputs['train'])

        if 'val' in phases:
            predicts = self.model_manager.predict(inputs['val'])
        else:
            predicts = None

        # save loss and metrics in one batch
        for metric in self.metrics[phase]:
            metric.update(loss, predicts, labels['val'])
            metric.average_meter.best_score = metric.average_meter.average

        for phase in phases:
            message += f'{phase} ['
            message += '\t'.join([f'{m.name}: {m.average_meter.average:.4f}' for m in self.metrics[phase]])
            message += ']\t'
        logger.info(message)

        return self.metrics, pred_list

    def train_with_early_stopping(self, model_manager=None, with_validate=True, only_validate=False) -> Tuple[Metrics, np.array]:
        if model_manager:
            self.model_manager = model_manager

        if with_validate:
            phases = ['train', 'val']
        else:
            phases = ['train']
        if only_validate:
            phases = ['val']

        self.check_keys_from_dict(phases, self.dataloaders)

        inputs, labels = {}, {}
        for phase in phases:
            inputs[phase] = np.vstack(tuple([data[0].numpy() for data in self.dataloaders[phase]]))
            labels[phase] = np.hstack(tuple([data[1].numpy() for data in self.dataloaders[phase]]))

        if 'train' in phases:
            loss, predicts = self.model_manager.fit(inputs['train'], labels['train'], inputs['val'], labels['val'])
            
        # save loss and metrics in one batch
        for metric in self.metrics[phase]:
            metric.update(loss, predicts, labels['val'])
            metric.average_meter.best_score = metric.average_meter.average

        message = 'val:' + '\t'.join([f'{m.name}: {m.average_meter.average:.4f}' for m in self.metrics['val']])
        message += ']\t'
        logger.info(message)

        return self.metrics, predicts
