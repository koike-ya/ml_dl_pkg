from torch.utils.data import DataLoader
import torch
import numpy as np


class WrapperDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(WrapperDataLoader, self).__init__(*args, **kwargs)


def make_weights_for_balanced_classes(labels, n_classes):
    labels = np.array(labels, dtype=int)
    class_count = [np.sum(labels == class_index) for class_index in range(n_classes)]
    weight_per_class = sum(class_count) / torch.Tensor(class_count)
    weights = [weight_per_class[label] for label in labels]
    return weights
