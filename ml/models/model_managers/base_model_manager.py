import logging
from abc import ABCMeta, abstractmethod

logger = logging.getLogger(__name__)

import numpy as np

from ml.models.ml_models.decision_trees import decision_tree_args
from ml.models.ml_models.toolbox import ml_model_manager_args
from ml.models.nn_models.adda import adda_args
from ml.models.nn_models.cnn import cnn_args
from ml.models.nn_models.rnn import rnn_args
from ml.models.nn_models.nn import nn_args
from ml.models.nn_models.pretrained_models import pretrain_args
from ml.models.loss import loss_args, set_criterion


# from ml.models.adda import adda_args


def model_args(parser):
    model_parser = parser.add_argument_group("Model arguments")
    # cnn|xgboost|knn|catboost|sgdc will be supported

    model_parser = parser.add_argument_group("ML/DL model arguments")
    model_parser.add_argument('--early-stopping', help='Early stopping with validation data', action='store_true')
    model_parser.add_argument('--return-prob', help='Returns probability', action='store_true')
    parser = nn_args(parser)
    parser = rnn_args(parser)
    parser = cnn_args(parser)
    parser = pretrain_args(parser)
    parser = adda_args(parser)
    parser = loss_args(parser)

    # ML系用のパラメータ
    parser = ml_model_manager_args(parser)
    parser = decision_tree_args(parser)

    return parser


from dataclasses import dataclass, field
from ml.models.nn_models.nn import NNConfig
from ml.models.nn_models.rnn import RNNConfig
from ml.models.nn_models.cnn import CNNConfig
from ml.models.nn_models.pretrained_models import PretrainedConfig
from ml.models.ml_models.toolbox import MlModelManagerConfig
from ml.models.ml_models.decision_trees import DecisionTreeConfig
from ml.models.loss import LossConfig

@dataclass
class ModelConfig(NNConfig, RNNConfig, CNNConfig, PretrainedConfig, LossConfig, MlModelManagerConfig, DecisionTreeConfig):      # ML/DL model arguments
# class ModelConfig:  # ML/DL model arguments
    early_stopping: bool = False        # Early stopping with validation data
    return_prob: bool = False     # Returns probability, not predicted labels
    nn_config: NNConfig = NNConfig()
    rnn_config: RNNConfig = RNNConfig()
    cnn_config: CNNConfig = CNNConfig()
    pretrained_config: PretrainedConfig = PretrainedConfig()
    loss_config: LossConfig = LossConfig()

    ml_model_manager_config: MlModelManagerConfig = MlModelManagerConfig()
    decision_tree_config: DecisionTreeConfig = DecisionTreeConfig()


class BaseModelManager(metaclass=ABCMeta):
    def __init__(self, class_labels, cfg, must_contain_keys):
        self.class_labels = class_labels
        self.cfg = self._check_cfg(cfg, must_contain_keys)
        self.criterion = set_criterion(self.cfg)
        self.fitted = False

    @staticmethod
    def _check_cfg(cfg, must_contain_keys):
        for key in must_contain_keys:
            assert key in cfg.keys(), f'{key} must be contained in the model conf'
        return cfg

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
