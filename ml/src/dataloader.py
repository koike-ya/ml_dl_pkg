from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from ml.utils.enums import TaskType


@dataclass
class DataConfig:
    batch_size: int = 32        # Batch size for training
    epoch_rate: float = 1.0     # Data rate to to use in one epoch
    task_type: TaskType = TaskType.classify
    n_jobs: int = 4             # Number of workers used in data-loading
    sample_balance: List[float] = field(default_factory=lambda: [])  # Sampling label balance from dataset
    tta: int = 0  # Number of test time augmentation ensemble


def set_dataloader(dataset, phase, cfg, shuffle=False, collate_fn=None):
    if phase != 'train':
        dataloader = WrapperDataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.n_jobs,
                                       pin_memory=False, sampler=None, shuffle=False, drop_last=False,
                                       collate_fn=collate_fn)
    else:
        if cfg.sample_balance:
            if cfg.task_type.value == 'classify':
                weights = make_weights_for_balanced_classes(dataset.get_labels(), cfg.sample_balance)
            else:
                weights = [torch.Tensor([1.0])] * len(dataset.get_labels())
            sampler = WeightedRandomSampler(weights, int(len(dataset) * cfg.epoch_rate))
        else:
            sampler = None
        dataloader = WrapperDataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.n_jobs,
                                       pin_memory=False, sampler=sampler, drop_last=True, shuffle=shuffle,
                                       collate_fn=collate_fn)
    return dataloader


def set_ml_dataloader(dataset, phase, cfg, shuffle=False):
    if phase in ['test', 'infer']:
        dataloader = WrapperDataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.n_jobs,
                                  pin_memory=True, sampler=None, shuffle=False, drop_last=False)
    else:
        if sum(cfg.sample_balance) != 0.0:
            if cfg.task_type.value == 'classify':
                weights = make_weights_for_balanced_classes(dataset.get_labels(), cfg.sample_balance)
            else:
                weights = [torch.Tensor([1.0])] * len(dataset.get_labels())
            sampler = WeightedRandomSampler(weights, int(len(dataset) * cfg.epoch_rate))
        else:
            sampler = None
        dataloader = WrapperDataLoader(dataset, batch_size=len(dataset), num_workers=cfg.n_jobs,
                                  pin_memory=True, sampler=sampler, shuffle=shuffle)
    return dataloader


def set_adda_dataloader(dataset, cfg):
    return WrapperDataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.n_jobs,
                             pin_memory=True, drop_last=True, shuffle=True)


class WrapperDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(WrapperDataLoader, self).__init__(*args, **kwargs)

    def get_input_size(self):
        return self.dataset.get_feature_size()

    def get_image_size(self):
        return self.dataset.get_image_size()

    def get_n_channels(self):
        return self.dataset.get_n_channels()

    def get_seq_len(self):
        return self.dataset.get_seq_len()


def make_weights_for_balanced_classes(labels, sample_balance):
    labels = np.array(labels, dtype=int)
    class_count = pd.Series(labels).value_counts(sort=False).values
    label_kind = list(set(labels))
    weight_per_class = sum(class_count) / torch.Tensor(class_count)
    if sample_balance == 'same':
        weights = [torch.Tensor([1.0])] * len(labels)
    else:
        weights = [weight_per_class[label_kind.index(label)] * sample_balance[label_kind.index(label)] for label in labels]

    return weights