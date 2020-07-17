from dataclasses import dataclass

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
class ExptConfig(BaseExptConfig):
    pass
