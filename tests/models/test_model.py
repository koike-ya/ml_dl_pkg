import unittest
from unittest import TestCase

import numpy as np
import torch

from ml.utils.config import TEST_PATH
from ml.models.model_managers.base_model_manager import BaseModelManager
from ml.models.model_managers.ml_model_manager import MLModel
from ml.models.model_managers.nn_model_manager import NNModelManager


class TestModel(TestCase):
    def setUp(self):
        self.class_labels = [0, 1, 2]
        batch_size = 64
        input_size = 3
        self.base_cfg = {
            'model_type': 'rnn', 'gpu_id': 0, 'optimizer': 'adam', 'lr': 0.001, 'momentum': 0.9, 'weight_decay': 0.0,
            'learning_anneal': 1.1, 'batch_size': 32, 'epoch_rate': 1.0, 'n_jobs': 4, 'loss_weight': [1.0, 1.0],
            'epochs': 2, 'model_path': 'outputs/models/sth.pth', 'seed': 0, 'silent': True,
            'log_id': 'results', 'tensorboard': False, 'log_dir': 'visualize/', 'max_norm': 400, 'rnn_type': 'gru',
            'rnn_hidden_size': 400, 'rnn_n_layers': 3, 'bidirectional': True, 'is_inference_softmax': True,
            'n_estimators': 2, 'max_depth': 3, 'reg_alpha': 0, 'reg_lambda': 1, 'subsample': 1.0
        }
        self.base_cfg['cuda'] = torch.cuda.is_available()
        self.base_cfg['input_size'] = input_size
        self.base_cfg['loss_weight'] = [1.0] * len(self.class_labels)
        self.base_cfg['task_type'] = 'classify'
        self.base_rnn_conf = {**self.base_cfg}
        self.base_cnn_conf = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sample_data = torch.Tensor([list(range(input_size)) + [0.0]] * batch_size).to(self.device)
        self.sample_data[0:3, 3] = torch.Tensor(list(range(len(self.class_labels))))  # Every label appears
        self.sample_data[0:3, 0] = 2.0      # For avoiding constant inputs
        self.ml_model_type_pattern = ['xgboost', 'knn', 'sgdc', 'catboost']
        self.rnn_type_pattern = ['lstm', 'gru', 'rnn']  # , 'deepspeech']

    def _make_common_test_pattern(self, test_method):
        test_pattern = []
        # TODO CNN版追加

        rnn_conf = self.base_rnn_conf.copy()
        rnn_conf['model_type'] = 'rnn'
        for rnn_type in self.rnn_type_pattern:
            rnn_conf['rnn_type'] = rnn_type
            test_pattern.append({
                'description': f'RNN_{rnn_type}の{test_method}',
                'model_cls': NNModelManager,
                'model_conf': {**rnn_conf},
            })

        model_conf = self.base_rnn_conf.copy()
        for model_type in self.ml_model_type_pattern:
            model_conf['model_type'] = model_type
            test_pattern.append({
                'description': f'{model_type}の{test_method}',
                'model_cls': MLModel,
                'model_conf': {**model_conf},
            })

        return test_pattern

    def test___init__(self):
        model = NNModelManager(self.class_labels, self.base_rnn_conf)
        self.assertEqual(self.class_labels, model.class_labels)
        with self.assertRaises(TypeError):
            model = BaseModelManager(self.class_labels, self.base_cfg)

    def test_model_args(self):
        model = NNModelManager(self.class_labels, self.base_cfg)

    def test__init_model(self):
        # TODO CNNとdeepspeechのテスト
        test_pattern = self._make_common_test_pattern('__init__')

        for test_case in test_pattern:
            with self.subTest(test_case['description']):
                model = test_case['model_cls'](self.class_labels, test_case['model_conf'])
                self.assertIsInstance(model, BaseModelManager)

    def test_fit(self):
        # TODO CNNとdeepspeechのテスト
        test_pattern = self._make_common_test_pattern('fit')

        phase_pattern = ['train', 'val']
        for test_case in test_pattern:
            model = test_case['model_cls'](self.class_labels, test_case['model_conf'])
            for phase in phase_pattern:
                loss, pred = model.fit(self.sample_data[:, :3], self.sample_data[:, 3], phase)
                with self.subTest(test_case['description'] + f' {phase}時'):
                    self.assertIsInstance(loss, float)
                    self.assertEqual(pred.shape[0], self.sample_data.shape[0])

    def test_save_model(self):
        test_pattern = self._make_common_test_pattern('save_model')

        model_path = TEST_PATH / 'test_saved_model_weight.pth'
        if model_path.is_file():
            model_path.unlink()
        for test_case in test_pattern:
            with self.subTest(test_case['description']):
                test_case['model_conf']['model_path'] = str(model_path)
                model = test_case['model_cls'](self.class_labels, test_case['model_conf'])
                model.save_model()
                self.assertTrue(model_path.is_file())
                model_path.unlink()

    def test_load_model(self):
        test_pattern = self._make_common_test_pattern('load_model')

        model_path = TEST_PATH / 'test_saved_model_weight.pth'
        for test_case in test_pattern:
            if model_path.is_file():
                model_path.unlink()
            test_case['model_conf']['model_path'] = str(model_path)
            model = test_case['model_cls'](self.class_labels, test_case['model_conf'])
            model.save_model()
            with self.subTest(test_case['description']):
                self.assertFalse(model.fitted)
                model.load_model()
                self.assertTrue(model.fitted)
            model_path.unlink()

    def test_predict(self):
        # np.arrayを返す
        # TODO fitしていない場合は学習済み重みを読み込む。保存済み重みがない場合のエラーハンドリング。
        # TODO CNNとdeepspeechのテスト
        test_pattern = self._make_common_test_pattern('predict')

        for test_case in test_pattern:
            model_path = TEST_PATH / 'test_saved_model_weight.pth'
            if model_path.is_file():
                model_path.unlink()
            test_case['model_conf']['model_path'] = str(model_path)

            # ベストモデルの読み込みなし
            model = test_case['model_cls'](self.class_labels, test_case['model_conf'])
            _ = model.fit(self.sample_data[:, :3], self.sample_data[:, 3], 'train')
            with self.subTest(test_case['description']):
                preds = model.predict(self.sample_data[:, :3])
                self.assertIsInstance(preds, np.ndarray)
                self.assertEqual(preds.shape[0], self.sample_data.shape[0])

            # ベストモデルの読み込みありで、エラーあり
            model = test_case['model_cls'](self.class_labels, test_case['model_conf'])
            with self.subTest(test_case['description']):
                _ = model.fit(self.sample_data[:, :3], self.sample_data[:, 3], 'train')
                with self.assertRaises(SystemExit):
                    model.load_model()

            # ベストモデルの読み込みありで、エラーなし
            model = test_case['model_cls'](self.class_labels, test_case['model_conf'])
            _ = model.fit(self.sample_data[:, :3], self.sample_data[:, 3], 'train')
            self.assertFalse(model_path.is_file())
            model.save_model()
            model.load_model()
            with self.subTest(test_case['description']):
                preds = model.predict(self.sample_data[:, :3])
                self.assertIsInstance(preds, np.ndarray)
                self.assertEqual(preds.shape[0], self.sample_data.shape[0])


if __name__ == '__main__':
    unittest.main()
