import logging
import os
import random
import time
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List, Union

import numpy as np
import pandas as pd
import torch
from ml.models.model_managers.base_model_manager import BaseModelManager
from ml.models.model_managers.base_model_manager import ExtendedModelConfig, ModelConfig
from ml.utils.enums import TaskType
from ml.utils.logger import TensorBoardLogger
from ml.utils.utils import Metrics
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)


@dataclass
class TensorboardConfig:
    log_id: str = 'results'         # Identifier for tensorboard run
    tensorboard: bool = False       # Turn on tensorboard graphing
    log_dir: str = '../visualize/tensorboard'   # Location of tensorboard log


@dataclass
class TrainConfig(TensorboardConfig):
    epochs: int = 70  # Number of Training Epochs
    task_type: TaskType = TaskType.classify
    cuda: bool = True  # Use cuda to train a model
    finetune: bool = False  # Fine-tune the model from checkpoint "continue_from"
    model: ModelConfig = ExtendedModelConfig()
    class_names: List[str] = field(default_factory=lambda: ['0', '1'])
    softlabel_t: float = 0.0

    train_path: str = 'input/train.csv'  # Data file for training
    val_path: str = 'input/val.csv'  # Data file for validation
    test_path: str = 'input/test.csv'  # Data file for testing

    gpu_id: int = 0  # ID of GPU to use

    tta: int = 0  # Number of test time augmentation ensemble
    snapshot: List[int] = field(
        default_factory=lambda: [])  # The number of epochs to save weights. Comma separated int is allowed
    cache: bool = False  # Make cache after preprocessing or not


@contextmanager
def simple_timer(label) -> None:
    start = time.time()
    yield
    end = time.time()
    logger.info('{}: {:.3f}'.format(label, end - start))


class BaseTrainManager(metaclass=ABCMeta):
    def __init__(self, class_labels, cfg, dataloaders, metrics):
        self.class_labels = class_labels
        self.cfg = cfg
        self.cfg.model.class_names = self.cfg.class_names
        self.cfg.model.task_type = self.cfg.task_type
        self.cfg.model.cuda = self.cfg.cuda
        self.dataloaders = dataloaders
        self.device = self._init_device()
        self.model_manager = self._init_model_manager()
        self.logger = self._init_logger()
        self.metrics = metrics
        Path(self.cfg.model.model_path).parent.mkdir(exist_ok=True, parents=True)

    @abstractmethod
    def _init_model_manager(self) -> BaseModelManager:
        pass

    def _init_device(self) -> torch.device:
        if self.cfg.cuda:# and self.cfg.model_type.value in [name.value for name in list(NNType) + list(PretrainedType)]:
            device = torch.device('cuda')
            torch.cuda.set_device(self.cfg['gpu_id'])
        else:
            device = torch.device('cpu')

        return device

    def _init_logger(self) -> TensorBoardLogger:
        if self.cfg['tensorboard']:
            return TensorBoardLogger(self.cfg['log_id'], self.cfg['log_dir'])

    def _record_log(self, phase, epoch, metrics, suffix='') -> None:
        values = {}

        for metric in metrics[phase]:
            values[f'{phase}_{metric.name}_mean{suffix}'] = metric.average_meter.average
        self.logger.update(epoch, values)

    def _predict(self, phase) -> Tuple[np.array, np.array]:
        raise NotImplementedError

    def _average_tta(self, pred_list, label_list):
        new_pred_list = np.zeros((-1, pred_list.shape[0] // self.cfg['tta']))
        for label in range(pred_list.shape[1]):
            new_pred_list[:, label] = pred_list[:, label].reshape(self.cfg['tta'], -1).mean(axis=0)
        pred_list = new_pred_list
        label_list = label_list[:label_list.shape[0] // self.cfg['tta']]

        return pred_list, label_list

    def train(self, model_manager=None, with_validate=True, only_validate=False) -> Tuple[Metrics, np.array]:
        raise NotImplementedError

    def test(self, return_metrics=False, load_best=True, phase='test') -> Union[Tuple[np.array, np.array, Metrics],
                                                                                Tuple[np.array, np.array]]:
        if load_best:
            self.model_manager.load_model()

        pred_list, label_list = self.predict(phase=phase)

        if self.cfg.model.return_prob:
            pred_onehot = torch.from_numpy(pred_list)
            pred_list = np.argmax(pred_list, axis=1)

        logger.debug(f'Prediction info:{pd.Series(pred_list).describe()}')

        for metric in self.metrics['test']:
            if metric.name == 'loss':
                if self.cfg['task_type'].value == 'classify':
                    y_onehot = torch.zeros(label_list.shape[0], len(self.class_labels))
                    y_onehot = y_onehot.scatter_(1, torch.from_numpy(label_list).view(-1, 1).type(torch.LongTensor), 1)
                    if not self.cfg.model.return_prob:
                        pred_onehot = torch.zeros(pred_list.shape[0], len(self.class_labels))
                        pred_onehot = pred_onehot.scatter_(1, torch.from_numpy(pred_list).view(-1, 1).type(torch.LongTensor), 1)
                    loss_value = self.model_manager.criterion(pred_onehot.to(self.device), y_onehot.to(self.device)).item()
                # elif self.cfg['model_type'] in ['rnn', 'cnn', 'cnn_rnn']:
                #     loss_value = self.model_manager.criterion(torch.from_numpy(pred_list).to(self.device),
                #                                               torch.from_numpy(label_list).to(self.device))
            else:
                loss_value = 10000000

            metric.update(loss_value=loss_value, preds=pred_list, labels=label_list)
            logger.info(f"{phase} {metric.name}: {metric.average_meter.value :.4f}")
            metric.average_meter.update_best()

        if self.cfg['task_type'].value == 'classify':
            confusion_matrix_ = confusion_matrix(label_list, pred_list,
                                                 labels=list(range(len(self.class_labels))))
            logger.info(f'Confusion matrix: \n{confusion_matrix_}')

        if return_metrics:
            if self.cfg.model.return_prob:
                return pred_onehot.numpy(), label_list, self.metrics
            else:
                return pred_list, label_list, self.metrics
        return pred_list, label_list

    def infer(self, load_best=True, phase='infer') -> np.array:
        if load_best:
            self.model_manager.load_model()

        pred_list, _ = self.predict(phase=phase)

        return pred_list
