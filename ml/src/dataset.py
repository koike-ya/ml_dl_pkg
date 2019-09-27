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


class CSVDataSet(BaseDataSet):
    def __init__(self, csv_path, data_conf, load_func=None, process_func=None):
        """
        data_conf: {
            'header': None or True,
            'feature_columns': list of feature columns,
            'label_column': column name to use as label
        }
        process_func gets np.array inputs, shape is (1, n_features) and puts out processed one

        """
        super(CSVDataSet, self).__init__()
        if load_func:
            df = load_func(csv_path)
        else:
            df = pd.read_csv(csv_path, header=data_conf.get('header', 'infer'))
        self.y = df.loc[:, data_conf.get('label_column', 'y')].values
        del df[data_conf.get('label_column', 'y')]
        self.x = df.values
        self.process_func = process_func if process_func else None

    def __getitem__(self, idx):
        # TODO infer時にyとしてList[None]を返す実装
        if self.process_func:
            return self.process_func(self.x[idx]), self.y[idx]
        return self.x[idx], self.y[idx]

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
    def __init__(self, manifest_path, data_conf, load_func=None, label_func=None, process_func=None):
        """
        data_conf: {
            'load_func': Function to load data from manifest correctly,
            'labels': List of labels or None if it's made from path
            'label_func': Function to extract labels from path or None if labels are given as 'labels'
        }

        """
        super(ManifestDataSet, self).__init__()
        self.path_list = list(pd.read_csv(manifest_path, header=None).values.reshape(-1,))
        self.load_func = load_func
        self.label_func = label_func
        self.labels = self._set_labels(data_conf['labels'] if 'labels' in data_conf.keys() else None)
        self.process_func = process_func if process_func else None

    def __getitem__(self, idx):
        # TODO phaseがinferの場合はlabelsは[None]で返す
        x = self.load_func(self.path_list[idx])
        label = self.labels[idx]

        if self.process_func:
            return self.process_func(x, label)

        return x, label

    def __len__(self):
        return len(self.path_list)

    def _set_labels(self, labels=None):
        if self.label_func:
            return [self.label_func(path) for path in self.path_list]
        else:
            return labels

    def get_feature_size(self):
        return self.load_func(self.path_list[0]).size(0)

    def get_labels(self):
        return self.labels
