from torch.utils.data import Dataset
import pandas as pd


class BaseDataSet(Dataset):
    def __init__(self):
        super(BaseDataSet, self).__init__()

    def get_feature_size(cls):
        pass


class CSVDataSet(BaseDataSet):
    def __init__(self, csv_path, data_conf):
        """
        data_conf: {
            'header': None or True,
            'feature_columns': list of feature columns,
            'label_column': column name to use as label
        }

        """
        super(CSVDataSet, self).__init__()
        df = pd.read_csv(csv_path, header=data_conf.get('header', 'infer'))
        self.x = df.loc[:, data_conf['feature_columns']].values
        self.y = df.loc[:, data_conf['label_column']].values

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.x.shape[0]

    def get_feature_size(self):
        return self.x.shape[1]

    def get_labels(self):
        return self.y


class ManifestDataSet(BaseDataSet):
    def __init__(self, manifest_path, data_conf):
        """
        data_conf: {
            'load_func': Function to load data from manifest correctly,
            'labels': List of labels or None if it's made from path
            'label_func': Function to extract labels from path or None if labels are given as 'labels'
        }

        """
        super(ManifestDataSet, self).__init__()
        self.path_list = pd.read_csv(manifest_path, header=None).values[0]
        self.load_func = data_conf['load_func']
        self.labels = data_conf['labels']   # self.labels is None in test dataset
        self.label_func = data_conf['label_func']

    def __getitem__(self, idx):
        x = self.load_func(self.path_list[idx])
        if self.label_func:
            labels = self.label_func(self.path_list[idx])
        else:
            labels = self.labels[idx]

        return x, labels

    def __len__(self):
        return len(self.path_list)

    def get_feature_size(self):
        return self.load_func(self.path_list[0]).size[1:]

    def get_labels(self):
        if self.label_func:
            return self.label_func(self.path_list)
        else:
            return self.labels
