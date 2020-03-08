import logging
import time

logger = logging.getLogger(__name__)

import numpy as np
from copy import deepcopy
from ml.models.model_managers.nn_model_manager import supported_nn_models, supported_pretrained_models
from ml.models.train_managers.train_manager import BaseTrainManager
from ml.models.model_managers.adda_model_manager import AddaModelManager
from ml.src.metrics import get_metrics
from tqdm import tqdm
from typing import Tuple

supported_ml_models = ['xgboost', 'knn', 'catboost', 'sgdc', 'lightgbm', 'svm']
supported_models = supported_ml_models + supported_nn_models + list(supported_pretrained_models.keys())


def loop_iterable(iterable):
    while True:
        yield from iterable


class AddaTrainManager(BaseTrainManager):
    def __init__(self, class_labels, cfg, dataloaders, metrics):
        super(AddaTrainManager, self).__init__(class_labels, cfg, dataloaders, metrics)
        self.metrics['source'] = get_metrics(['loss'], target_metric=False)
        self.metrics['target'] = get_metrics(['loss'], target_metric=False)

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

    def train(self, model_manager=None, with_validate=True, only_validate=False) -> None:

        super().train(with_validate=with_validate)

        start = time.time()
        epoch_metrics = {}
        best_val_pred = np.array([])
        batch_size = self.cfg['batch_size']

        for epoch in range(self.cfg['adda_epochs']):

            phase = 'source'
            batch_iterator = zip(loop_iterable(self.dataloaders['source']), loop_iterable(self.dataloaders['target']))
            # Train discriminator
            for i in range(self.cfg['k_disc']):
                (src_inputs, _), (tgt_inputs, _) = next(batch_iterator)
                loss = self.model_manager.fit_discriminator(src_inputs.to(self.device), tgt_inputs.to(self.device))
                # save loss and metrics in one batch
                for metric in self.metrics[phase]:
                    metric.update(loss, _, _)
                self._verbose(epoch, phase, i, elapsed=int(time.time() - start), data_len=self.cfg['k_disc'])
            logger.info(f"Epoch {epoch} Discriminator loss: {self.metrics[phase][0].average_meter.average}")
            if self.logger:
                self._record_log(phase, epoch)
            epoch_metrics[phase] = deepcopy(self.metrics[phase])

            phase = 'target'
            batch_iterator = loop_iterable(self.dataloaders['target'])
            # Train classifier
            for i in range(self.cfg['k_clf']):
                (tgt_inputs, _) = next(batch_iterator)
                loss = self.model_manager.fit_classifier(tgt_inputs.to(self.device))
                # save loss and metrics in one batch
                for metric in self.metrics[phase]:
                    metric.update(loss, _, _)
                self._verbose(epoch, phase, i, elapsed=int(time.time() - start), data_len=self.cfg['k_clf'])
            logger.debug(f"Epoch {epoch} Discriminator loss: {self.metrics[phase][0].average_meter.average}")
            if self.logger:
                self._record_log(phase, epoch)
            epoch_metrics[phase] = deepcopy(self.metrics[phase])

            self.model_manager.model.feature_extractor = self.model_manager.tgt.feature_extractor

            if with_validate:
                phase = 'val'
                # Validate with class label
                orig_epochs = self.cfg['epochs']
                self.cfg['epochs'] = 1
                metrics, pred_list = super().train(only_validate=True)
                epoch_metrics[phase] = deepcopy(metrics[phase])
                best_val_flag = self._update_by_epoch(phase, epoch, self.cfg['learning_anneal'])
                if best_val_flag:
                    best_val_pred = pred_list
                self.cfg['epochs'] = orig_epochs

            self._epoch_verbose(epoch, epoch_metrics, ['source', 'target'])

        if self.logger:
            self.logger.close()

        return self.metrics, best_val_pred

    def infer(self, load_best=True, phase='infer') -> np.array:
        if load_best:
            self.model.load_model()

        pred_list, _ = self._predict(phase=phase)

        return pred_list
