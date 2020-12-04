from dataclasses import dataclass, field
from typing import List, Any

from ml.models.loss import LossConfig
from ml.models.model_managers.base_model_manager import ModelConfig
from ml.models.nn_models.attention import AttnConfig
from ml.models.nn_models.multitask_predictor import MultitaskConfig
from omegaconf import MISSING


@dataclass
class OptimConfig:
    lr: float = 1e-3  # Initial learning rate
    learning_anneal: float = 1.1  # Annealing applied to learning rate after each epoch
    weight_decay: float = 1e-5  # Initial Weight Decay


@dataclass
class SGDConfig(OptimConfig):
    momentum: float = 0.9


@dataclass
class AdamConfig(OptimConfig):
    eps: float = 1e-8  # Adam eps
    betas: tuple = (0.9, 0.999)  # Adam betas


@dataclass
class StackedModelConfig(AttnConfig, MultitaskConfig):
    pass


@dataclass
class NNModelConfig(ModelConfig, StackedModelConfig):
    image_size: List[int] = field(default_factory=lambda: [])
    in_channels: int = 0
    attention: bool = False
    grad_clip: List[float] = field(default_factory=lambda: [])
    loss_config: LossConfig = LossConfig()
    checkpoint_path: str = ''  # Model weight file to load model
    amp: bool = False  # Mixed precision training

    cuda: bool = True  # Use cuda to train a model
    optim: Any = MISSING
