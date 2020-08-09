from dataclasses import dataclass, field
from typing import List

from ml.models.model_managers.base_model_manager import ModelConfig


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
class NNModelConfig(ModelConfig):
    image_size: List[int] = field(default_factory=lambda: [])
    in_channels: int = 0
