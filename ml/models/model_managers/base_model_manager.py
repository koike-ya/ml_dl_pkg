import logging
from abc import ABCMeta, abstractmethod

logger = logging.getLogger(__name__)

import numpy as np

from ml.models.loss import set_criterion


from dataclasses import dataclass, field
from ml.models.loss import LossConfig
from typing import List, Any
from omegaconf import MISSING
from ml.utils.enums import TaskType, ModelType
from ml.preprocess.augment import SpecAugConfig


@dataclass
class ModelConfig:  # ML/DL model arguments
    model_name: str = ''
    model_path: str = '../output/models/sth.pth'    # Path to save model
    early_stopping: bool = False        # Early stopping with validation data
    return_prob: bool = False     # Returns probability, not predicted labels
    loss_config: LossConfig = LossConfig()
    checkpoint_path: str = ''  # Model weight file to load model
    amp: bool = False  # Mixed precision training

    input_size: List[int] = field(default_factory=lambda: [])

    # TODO train_managerと共有したままなのか、継承によって消すのか
    class_names: List[str] = field(default_factory=lambda: ['0', '1'])
    task_type: TaskType = TaskType.classify
    cuda: bool = True  # Use cuda to train a model
    transfer: bool = False  # TODO modify this or remove this feature # Transfer learning from model_path
    model_type: ModelType = ModelType.cnn

    optim: Any = MISSING


@dataclass
class ExtendedModelConfig(ModelConfig):
    mixup_alpha: float = 0.0    # Beta distirbution alpha for mixup
    spec_augment: SpecAugConfig = SpecAugConfig()


class BaseModelManager(metaclass=ABCMeta):
    def __init__(self, class_labels, cfg):
        self.class_labels = class_labels
        self.cfg = cfg
        self.criterion = set_criterion(self.cfg.loss_config, self.cfg.task_type.value, self.cfg.class_names)
        self.fitted = False

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
