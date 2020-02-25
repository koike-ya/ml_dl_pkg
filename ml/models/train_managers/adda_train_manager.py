import logging
import time

logger = logging.getLogger(__name__)

import numpy as np
import torch
from copy import deepcopy
from ml.models.model_managers.nn_model_manager import supported_nn_models, supported_pretrained_models
from ml.models.train_managers.train_manager import BaseTrainManager
from ml.models.model_managers.adda_model_manager import AddaModelManager
from ml.models.nn_models.nn_utils import set_requires_grad
from sklearn.metrics import confusion_matrix
from torch import nn
from tqdm import tqdm
from typing import Tuple, Union, List
from ml.utils.utils import Metrics

supported_ml_models = ['xgboost', 'knn', 'catboost', 'sgdc', 'lightgbm', 'svm']
supported_models = supported_ml_models + supported_nn_models + list(supported_pretrained_models.keys())


def loop_iterable(iterable):
    while True:
        yield from iterable


class AddaTrainManager(BaseTrainManager):
    def __init__(self, class_labels, cfg, dataloaders, metrics):
        super(AddaTrainManager, self).__init__(class_labels, cfg, dataloaders, metrics)

    def _init_model_manager(self) -> AddaModelManager:
        return AddaModelManager(self.class_labels, self.cfg)

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

    def train(self, model=None, with_validate=True, only_validate=False) -> None:

        super().train(with_validate=with_validate)

        start = time.time()
        epoch_metrics = {}
        best_val_pred = np.array([])

        for epoch in range(self.cfg['adda_epochs']):
            batch_iterator = zip(loop_iterable(self.dataloaders['source']), loop_iterable(self.dataloaders['target']))
            # Train discriminator
            disc_loss = 0
            for _ in range(self.cfg['k_disc']):
                (src_inputs, _), (tgt_inputs, _) = next(batch_iterator)
                disc_loss += self.model_manager.fit_discriminator(src_inputs.to(self.device), tgt_inputs.to(self.device))
            logger.debug(f'Epoch {epoch} Discriminator loss: {disc_loss}')

            batch_iterator = loop_iterable(self.dataloaders['target'])
            # Train classifier
            for _ in range(self.cfg['k_clf']):
                (tgt_inputs, _) = next(batch_iterator)
                feature_extractor = self.model_manager.fit_classifier(tgt_inputs.to(self.device))
            logger.debug(f'Epoch {epoch} Discriminator loss: {disc_loss}')

            self.model.feature_extractor = self.model_manager.tgt.feature_extractor

            if with_validate:
                # Validate with class label
                orig_epochs = self.cfg['epochs']
                self.cfg['epochs'] = 1
                metrics, pred = super().train(only_validate=True)
                self.cfg['epochs'] = orig_epochs

        if self.logger:
            self.logger.close()

        return self.metrics, best_val_pred

    def infer(self, load_best=True, phase='infer') -> np.array:
        if load_best:
            self.model.load_model()

        pred_list, _ = self._predict(phase=phase)

        return pred_list
