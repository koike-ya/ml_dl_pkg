import logging
from abc import ABCMeta, abstractmethod

import pandas as pd
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BaseDataSet(Dataset, metaclass=ABCMeta):
    def __init__(self):
        super(BaseDataSet, self).__init__()

    @abstractmethod
    def get_feature_size(self):
        pass

    @abstractmethod
    def get_labels(self):
        pass


class SimpleCSVDataset(BaseDataSet):
    def __init__(self, df, y, phase):
        super(SimpleCSVDataset, self).__init__()
        self.x = df.values
        self.y = y.values
        self.phase = phase

    def __getitem__(self, idx):
        if self.phase in ['train', 'val', 'test']:
            return self.x[idx].reshape(1, -1), self.y[idx]
        else:
            return self.x[idx]

    def get_feature_size(self):
        return self.x.shape[1]

    def __len__(self):
        return self.x.shape[0]

    def get_labels(self):
        return self.y

    def get_n_channels(self):
        return 1

    def get_image_size(self):
        return (self.x.shape[1],)


class CSVDataSet(BaseDataSet):
    def __init__(self, csv_path, data_conf, phase, load_func=None, transform=None, label_func=None):
        """
        data_conf: {
            'header': None or True,
            'feature_columns': list of feature columns,
            'label_column': column name to use as label
        }
        transform gets np.array inputs, shape is (1, n_features) and puts out processed one

        """
        super(CSVDataSet, self).__init__()
        self.phase = phase

        if load_func:
            self.x, self.y = load_func(csv_path)
            logger.debug(f'{phase}: mean {self.x.mean()}\t std {self.x.std()}')
        else:
            df = pd.read_csv(csv_path, header=data_conf.get('header', 'infer'))
            if phase in ['train', 'val']:
                self.y = df.iloc[:, -1]
                self.x = df.iloc[:, :-1].values

        self.transform = transform

    def __getitem__(self, idx):
        # TODO infer時にyとしてList[None]を返す実装
        if self.transform:
            return self.transform(self.x[idx]), self.y[idx]

        if self.phase in ['train', 'val']:
            return self.x[idx], self.y[idx]
        else:
            return self.x[idx]

    def __len__(self):
        return self.x.shape[0]

    def get_feature_size(self):
        if self.transform:
            return self.transform(self.x[0]).shape[0]
        else:
            return self.x.shape[1]

    def get_labels(self):
        return self.y

    def get_n_channels(self):
        return 1

    def get_image_size(self):
        return (self.x.shape[1],)

    def get_seq_len(self):
        return 32


class ManifestDataSet(BaseDataSet):
    # TODO 要テスト実装
    def __init__(self, manifest_path, data_conf, phase='train', load_func=None, transform=None, label_func=None):
        """
        data_conf: {
            'load_func': Function to load data from manifest correctly,
            'labels': List of labels or None if it's made from path
            'label_func': Function to extract labels from path or None if labels are given as 'labels'
        }

        """
        super(ManifestDataSet, self).__init__()
        self.path_df = pd.read_csv(manifest_path, header=None)
        if phase == 'test' and data_conf['tta']:
            self.path_df = pd.concat([self.path_df] * data_conf['tta'])
        self.load_func = load_func
        self.label_func = label_func
        if phase == 'infer':
            self.labels = [-100] * len(self.path_df)
        else:
            self.labels = self._set_labels(data_conf['labels'] if 'labels' in data_conf.keys() else None)
        self.transform = transform
        self.phase = phase

    def __getitem__(self, idx):
        x = self.load_func(self.path_df.iloc[idx, :])
        label = self.labels[idx]

        if self.transform:
            return self.transform(x), label

        return x, label

    def __len__(self):
        return self.path_df.shape[0]

    def _set_labels(self, labels=None):
        if self.label_func:
            return self.path_df.apply(self.label_func, axis=1)
        else:
            return labels

    def get_feature_size(self):
        x = self.load_func(self.path_df.iloc[0, :])
        if self.transform:
            x = self.transform(x)
        return x.size()

    def get_labels(self):
        return self.labels

    def get_image_size(self):
        return self.get_feature_size()[1:]

    def get_n_channels(self):
        return self.get_feature_size()[0]


class ManifestWaveDataSet(ManifestDataSet):
    def __init__(self, manifest_path, data_conf, load_func=None, transform=None, label_func=None, phase='train'):
        super(ManifestWaveDataSet, self).__init__(manifest_path, data_conf, load_func, transform, label_func, phase)

    def get_seq_len(self):
        x = self.load_func(self.path_df.iloc[0, :])
        if self.transform:
            x = self.transform(x)
        return x.size(1)
