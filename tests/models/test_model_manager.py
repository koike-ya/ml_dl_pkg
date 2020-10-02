import unittest

import torch
from ml.models.train_managers.train_manager import BaseTrainManager
from torch.utils.data import DataLoader

from ml.utils.config import TEST_PATH
from ml.src.dataset import BaseDataSet
from ml.src.metrics import Metric


class MockDataSet(BaseDataSet):
    def __init__(self, batch_size, input_size):
        super(MockDataSet, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size

    def __getitem__(self, idx):
        mock_test_x = torch.Tensor(list(range(self.input_size)))
        mock_test_y = torch.Tensor([0])
        return mock_test_x, mock_test_y

    def __len__(self):
        return self.batch_size * 3 + 2

    def get_feature_size(self):
        return self.input_size

    def get_labels(self):
        return torch.Tensor([0]) * (self.batch_size * 3 + 2)


class MockDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(MockDataLoader, self).__init__(*args, **kwargs)
        self.feature_size = self.dataset.get_feature_size()


class TestTrainManager(unittest.TestCase):

    def setUp(self):
        # TODO GPU/CPUの互換性テストをどうやるか
        self.base_classes = [0, 1, 2]
        self.input_size = 3
        self.base_cfg = {
            'model_type': 'rnn', 'gpu_id': 0, 'optimizer': 'adam', 'lr': 0.001, 'momentum': 0.9, 'weight_decay': 0.0,
            'learning_anneal': 1.1, 'batch_size': 32, 'epoch_rate': 1.0, 'num_workers': 4, 'loss_weight': [1.0, 1.0],
            'epochs': 2, 'model_path': f'{TEST_PATH}/outputs/models/sth.pth', 'seed': 0, 'silent': True,
            'log_id': 'results', 'tensorboard': False, 'log_dir': 'visualize/', 'max_norm': 400, 'rnn_type': 'gru',
            'rnn_hidden_size': 400, 'rnn_n_layers': 3, 'bidirectional': True, 'is_inference_softmax': True
        }
        self.base_cfg['cuda'] = torch.cuda.is_available()
        self.base_cfg['loss_weight'] = [1.0] * len(self.base_classes)
        self.base_cfg['input_size'] = self.input_size
        self.base_cfg['task_type'] = 'classify'     # TODO regresのテストも実装
        self.dataloaders = {}
        for phase in ['train', 'val']:
            dataset = MockDataSet(self.base_cfg['batch_size'], self.input_size)
            self.dataloaders[phase] = MockDataLoader(dataset, batch_size=self.base_cfg['batch_size'])
        self.metrics = [
            Metric('loss', direction='minimize'),
            Metric('accuracy', direction='maximize', save_model=True),
        ]

    def tearDown(self):
        pass

    def test_model_args(self):
        must_contain_keys = ['model_path', 'silent']
        for key in must_contain_keys:
            # print(key, key in args_dict.keys())
            self.assertTrue(key in self.base_cfg.keys())

    def test_BaseTrainManager(self):

        train_manager = BaseTrainManager(self.base_classes, self.base_cfg, self.dataloaders, self.metrics)
        self.assertEqual(self.base_classes, train_manager.class_labels)

    def test_train(self):
        self.base_cfg['epochs'] = 1
        train_manager = BaseTrainManager(self.base_classes, self.base_cfg, self.dataloaders, self.metrics)
        model = train_manager.train()
        self.assertIsInstance(model.model, torch.nn.Module)

    def test_test(self):
        phase = 'test'
        dataset = MockDataSet(self.base_cfg['batch_size'], self.input_size)
        dataloaders = {**self.dataloaders}
        dataloaders.update({phase: MockDataLoader(dataset, batch_size=self.base_cfg['batch_size'])})
        train_manager = BaseTrainManager(self.base_classes, self.base_cfg, dataloaders, self.metrics)
        train_manager.train()
        preds = train_manager.test()
        self.assertTrue(len(dataloaders['test']), len(preds))

    def test__predict(self):
        # TODO np.arrayで格納するのか、tensorで格納するのか決定する。現在はtensorのみ。
        # TODO 上のときNNModelManagerまたはMLModelのどちらかのpredictも変更する
        pass


if __name__ == '__main__':
    unittest.main()
