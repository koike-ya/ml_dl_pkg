import logging
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from typing import List

import numpy as np
from ml.models.loss import set_criterion
from ml.utils import init_seed
from ml.utils.enums import TaskType, MilType, MilAggType

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:  # ML/DL model arguments
    model_name: str = ''
    model_path: str = '../output/models/sth.pth'    # Path to save model
    early_stopping: bool = False        # Early stopping with validation data
    return_prob: bool = False     # Returns probability, not predicted labels
    seed: int = 0  # Seed for deterministic
    models: List[str] = field(default_factory=lambda: ['cnn'])
    input_size: List[int] = field(default_factory=lambda: [])

    # TODO train_managerと共有したままなのか、継承によって消すのか
    class_names: List[str] = field(default_factory=lambda: ['0', '1'])
    task_type: TaskType = TaskType.classify


@dataclass
class MilConfig:    # Multiple instance learning config
    train_mil: bool = False
    mil_finetune: bool = True
    mil_type: MilType = 'instance'            # Aggregation type
    mil_agg_func: MilAggType = 'mean'           # Aggregation function if mil_type in ['instance', 'embedding']


@dataclass
class ExtendedModelConfig(ModelConfig, MilConfig):
    mixup_alpha: float = 0.0    # Alpha of beta distirbution for mixup


class BaseModelManager(metaclass=ABCMeta):
    def __init__(self, class_labels, cfg):
        self.class_labels = class_labels
        self.cfg = cfg
        self.criterion = set_criterion(self.cfg.loss_config, self.cfg.task_type.value, self.cfg.class_names)
        self.fitted = False
        init_seed(self.cfg.seed)

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
