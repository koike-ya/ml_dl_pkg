from dataclasses import dataclass, field
from typing import Any, List

from omegaconf import MISSING
from ml.tasks.base_experiment import BaseExptConfig


@dataclass
class DataConfig:
    train_manifest: str = 'data/train_manifest.csv'
    val_manifest: str = 'data/val_manifest.csv'
    batch_size: int = 20  # Batch size for training
    num_workers: int = 4  # Number of workers used in data-loading
    labels_path: str = 'labels.json'  # Contains tokens for model output
    # spect: SpectConfig = SpectConfig()
    # augmentation: AugmentationConfig = AugmentationConfig()


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
class ExptConfig(BaseExptConfig):
    pass
