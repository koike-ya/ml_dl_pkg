import unittest
from unittest import TestCase

import numpy as np
import torch
from sklearn.exceptions import NotFittedError
from xgboost.core import XGBoostError

from ml.utils.config import TEST_PATH
from ml.models.ml_models.decision_trees import XGBoost, CatBoost
from ml.models.ml_models.toolbox import BaseMLPredictor, KNN, SGDC


class TestModel(TestCase):
    def setUp(self):
        self.class_labels = [0, 1, 2]
        batch_size = 64
        input_size = 3
        self.sample_data = torch.Tensor([list(range(input_size)) + [0.0]] * batch_size)
        self.sample_data[0:3, 3] = torch.Tensor(list(range(len(self.class_labels))))  # Every label appears
        self.supported_model = [KNN, SGDC, XGBoost, CatBoost]
        self.model_names = ['knn', 'sgdc', 'xgboost']
        self.base_cfg = {
            'lr': 0.001, 'n_jobs': 4, 'loss_weight': [1.0, 1.0, 1.0], 'model_path': 'outputs/models/sth.pth',
            'seed': 0, 'n_estimators': 2, 'max_depth': 3, 'reg_alpha': 0, 'reg_lambda': 1, 'subsample': 1.0
        }
        self.base_cfg['task_type'] = 'classify'  # TODO regressも実装

    def test___init__(self):
        test_pattern = []
        for model_name, model in zip(self.model_names, self.supported_model):
            test_pattern.append({
                'description': f'{model_name} init',
                'model_cls': model
            })
        for test_case in test_pattern:
            with self.subTest(test_case['description']):
                self.assertIsInstance(test_case['model_cls'](self.class_labels, self.base_cfg), BaseMLPredictor)

    def test_partial_fit(self):
        test_pattern = []
        for model_name, model in zip(self.model_names, self.supported_model):
            test_pattern.append({
                'description': f'{model_name} partial fit',
                'model_cls': model
            })
        for test_case in test_pattern:
            with self.subTest(test_case['description']):
                model = test_case['model_cls'](self.class_labels, self.base_cfg)
                loss = model.partial_fit(self.sample_data[:, :3], self.sample_data[:, 3])
                self.assertIsInstance(loss, np.float)

    def test_predict(self):
        test_pattern = []
        not_fitted_error_list = [NotFittedError, NotFittedError, XGBoostError]
        for model_name, model, error in zip(self.model_names, self.supported_model, not_fitted_error_list):
            test_pattern.append({
                'description': f'{model_name} predict',
                'model_cls': model,
                'expected_error': error
            })
        for test_case in test_pattern:
            with self.subTest(test_case['description'] + ' with fit'):  # fitしてからパターン
                model = test_case['model_cls'](self.class_labels, self.base_cfg)
                _ = model.partial_fit(self.sample_data[:, :3], self.sample_data[:, 3])
                preds = model.predict(self.sample_data[:, :3])
                self.assertEqual(self.sample_data.size(0), preds.shape[0])
            with self.subTest(test_case['description'] + ' without fit'):  # fitしないパターン
                model = test_case['model_cls'](self.class_labels, self.base_cfg)
                with self.assertRaises(test_case['expected_error']):
                    preds = model.predict(self.sample_data[:, :3])

    def test_predict_proba(self):
        test_pattern = []
        not_fitted_error_list = [NotFittedError, NotFittedError, XGBoostError]
        for model_name, model, error in zip(self.model_names, self.supported_model, not_fitted_error_list):
            test_pattern.append({
                'description': f'{model_name} predict_proba',
                'model_cls': model,
                'expected_error': error
            })
        for test_case in test_pattern:
            with self.subTest(test_case['description'] + ' with fit'):  # fitしてからパターン
                model = test_case['model_cls'](self.class_labels, self.base_cfg)
                _ = model.partial_fit(self.sample_data[:, :3], self.sample_data[:, 3])
                pred_probas = model.predict_proba(self.sample_data[:, :3])
                self.assertEqual(self.sample_data.size(0), pred_probas.shape[0])
                self.assertEqual(len(self.class_labels), pred_probas.shape[1])
            with self.subTest(test_case['description'] + ' without fit'):  # fitしないパターン
                model = test_case['model_cls'](self.class_labels, self.base_cfg)
                with self.assertRaises(test_case['expected_error']):
                    pred_probas = model.predict_proba(self.sample_data[:, :3])

    def test_save_model(self):
        test_pattern = []
        for model_name, model in zip(self.model_names, self.supported_model):
            test_pattern.append({
                'description': f'{model_name} save model',
                'model_cls': model,
            })
        model_path = TEST_PATH / 'test_saved_model_weight.pth'
        for test_case in test_pattern:
            with self.subTest(test_case['description']):
                model = test_case['model_cls'](self.class_labels, self.base_cfg)
                model.save_model(model_path)
                self.assertTrue(model_path.is_file())
                model_path.unlink()

    def test_load_model(self):
        test_pattern = []
        for model_name, model in zip(self.model_names, self.supported_model):
            test_pattern.append({
                'description': f'{model_name} load model',
                'model_cls': model,
            })
        model_path = TEST_PATH / 'test_saved_model_weight.pth'
        for test_case in test_pattern:
            self.assertFalse(model_path.is_file())
            model = test_case['model_cls'](self.class_labels, self.base_cfg)
            model.save_model(model_path)
            with self.subTest(test_case['description']):
                self.assertFalse(model.fitted)
                model.load_model(model_path)
                self.assertTrue(model.fitted)
            model_path.unlink()


if __name__ == '__main__':
    unittest.main()
