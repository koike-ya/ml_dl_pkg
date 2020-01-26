from abc import ABCMeta, abstractmethod

import pandas as pd
from torch.utils.data import Dataset


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
    def __init__(self, csv_path, data_conf, phase, load_func=None, process_func=None, label_func=None):
        """
        data_conf: {
            'header': None or True,
            'feature_columns': list of feature columns,
            'label_column': column name to use as label
        }
        process_func gets np.array inputs, shape is (1, n_features) and puts out processed one

        """
        super(CSVDataSet, self).__init__()
        self.phase = phase

        if load_func:
            df = load_func(csv_path)
        else:
            df = pd.read_csv(csv_path, header=data_conf.get('header', 'infer'))

        # TODO yの指定を修正。Manifest側のheaderない問題とうまいこと。
        if phase in ['train', 'val']:
            self.y = df.iloc[:, -1]
            self.x = df.iloc[:, :-1].values
        else:
            self.x = df.values
        self.process_func = process_func if process_func else None

    def __getitem__(self, idx):
        # TODO infer時にyとしてList[None]を返す実装
        if self.process_func:
            return self.process_func(self.x[idx]), self.y[idx]

        if self.phase in ['train', 'val']:
            return self.x[idx], self.y[idx]
        else:
            return self.x[idx]

    def __len__(self):
        return self.x.shape[0]

    def get_feature_size(self):
        if self.process_func:
            return self.process_func(self.x[0]).shape[0]
        else:
            return self.x.shape[1]

    def get_labels(self):
        return self.y


class ManifestDataSet(BaseDataSet):
    # TODO 要テスト実装
    def __init__(self, manifest_path, data_conf, load_func=None, process_func=None, label_func=None, phase='train'):
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
        self.labels = self._set_labels(data_conf['labels'] if 'labels' in data_conf.keys() else None)
        self.process_func = process_func if process_func else None
        self.phase = phase

    def __getitem__(self, idx):
        # TODO phaseがinferの場合はlabelsは[None]で返す
        x = self.load_func(self.path_df.iloc[idx, :])
        label = self.labels[idx]

        if self.process_func:
            return self.process_func(x, label)

        return x, label

    def __len__(self):
        return self.path_df.shape[0]

    def _set_labels(self, labels=None):
        if self.label_func:
            return [self.label_func(row) for i, row in self.path_df.iterrows()]
        else:
            return labels

    def get_feature_size(self):
        x = self.load_func(self.path_df.iloc[0, :])
        if self.process_func:
            x, _ = self.process_func(x, 0)
        return x.size()

    def get_labels(self):
        return self.labels
