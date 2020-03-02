import logging
from abc import ABCMeta, abstractmethod

logger = logging.getLogger(__name__)

import numpy as np
import torch

from ml.models.ml_models.decision_trees import decision_tree_args
from ml.models.ml_models.toolbox import ml_model_manager_args
from ml.models.nn_models.adda import adda_args
from ml.models.nn_models.cnn import cnn_args
from ml.models.nn_models.rnn import rnn_args
from ml.models.nn_models.nn import nn_args


# from ml.models.adda import adda_args


def model_args(parser):
    model_parser = parser.add_argument_group("Model arguments")
    # cnn|xgboost|knn|catboost|sgdc will be supported

    nn_parser = parser.add_argument_group("Neural nerwork model arguments")
    nn_parser.add_argument('--early-stopping', help='Early stopping with validation data', action='store_true')
    parser = nn_args(parser)
    parser = rnn_args(parser)
    parser = cnn_args(parser)
    parser = adda_args(parser)

    # ML系用のパラメータ
    parser = ml_model_manager_args(parser)
    parser = decision_tree_args(parser)

    return parser


class BaseModelManager(metaclass=ABCMeta):
    def __init__(self, class_labels, cfg, must_contain_keys):
        self.class_labels = class_labels
        self.cfg = self._check_cfg(cfg, must_contain_keys)
        self.criterion = self._set_criterion()
        self.fitted = False

    @staticmethod
    def _check_cfg(cfg, must_contain_keys):
        for key in must_contain_keys:
            assert key in cfg.keys(), f'{key} must be contained in the model conf'
        return cfg

    def _set_criterion(self):
        if isinstance(self.cfg['loss_weight'], str):
            self.cfg['loss_weight'] = [1.0] * len(self.cfg['class_names'])
        if self.cfg['task_type'] == 'regress':
            return torch.nn.MSELoss()
        else:
            assert len(self.class_labels) == len(self.cfg['loss_weight']), \
                'loss weight needs to be matched with the number of classes'
            return torch.nn.BCELoss(weight=torch.tensor(self.cfg['loss_weight']))

    def anneal_lr(self, learning_anneal):
        pass

    @abstractmethod
    def fit(self, inputs, labels, phase):
        pass

    def save_model(self):
        logger.info(f"Best model is saved to {self.cfg['model_path']}")
        self.model.save_model(self.cfg['model_path'])

    def load_model(self):
        # MLModelは各Modelがfittedを管理しているため、エラーハンドリングの必要がない
        try:
            self.model.load_model(self.cfg['model_path'])
            print('Saved model loaded.')
        except FileNotFoundError as e:
            print(e)
            print(f"trained model file doesn't exist at {self.cfg['model_path']}")
            exit(1)

        self.fitted = self.model.fitted

    @abstractmethod
    def predict(self, inputs) -> np.array:
        pass

    def update_by_epoch(self, phase):
        pass
