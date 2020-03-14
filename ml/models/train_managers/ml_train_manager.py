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
        inputs = np.vstack(tuple([x.numpy() for x, _ in self.dataloaders[phase]]))
        label_list = np.hstack(tuple([y for _, y in self.dataloaders[phase]]))

        pred_list = self.model_manager.predict(inputs)

        if self.cfg['tta']:
            pred_list, label_list = self._average_tta(pred_list, label_list)

        return pred_list, label_list

    def train(self, model_manager=None, with_validate=True, only_validate=False) -> Tuple[Metrics, np.array]:
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
            inputs[phase] = np.vstack(tuple([x.numpy() for x, _ in self.dataloaders[phase]]))
            labels[phase] = np.hstack(tuple([y for _, y in self.dataloaders[phase]]))

        if 'train' in phases:
            if self.cfg['early_stopping'] and 'val' in phases:
                loss = self.model_manager.fit(inputs['train'], labels['train'], inputs['val'], labels['val'])
            else:
                loss = self.model_manager.fit(inputs['train'], labels['train'])

            predicts = {}
            for phase in phases:
                predicts[phase] = self.model_manager.predict(inputs[phase])

        for phase in phases:
            # save loss and metrics in one batch
            for metric in self.metrics[phase]:
                metric.update(loss, predicts[phase], labels[phase])
                metric.average_meter.best_score = metric.average_meter.average

        message = ''
        for phase in phases:
            message += f'{phase} ['
            message += '\t'.join([f'{m.name}: {m.average_meter.average:.4f}' for m in self.metrics[phase]])
            message += ']\t'
        logger.info(message)

        self.model_manager.save_model()

        return self.metrics, list(predicts.values())[-1]
