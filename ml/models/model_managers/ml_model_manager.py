import numpy as np
import torch

from ml.models.ml_models.decision_trees import CatBoost, XGBoost, LightGBM, RandomForest
from ml.models.ml_models.toolbox import KNN, SGDC, SVM, NaiveBayes
from ml.models.model_managers.base_model_manager import BaseModelManager

supported_ml_models = ['xgboost', 'knn', 'catboost', 'sgdc', 'lightgbm', 'svm', 'rf', 'nb']


class MLModelManager(BaseModelManager):
    def __init__(self, class_labels, cfg):
        must_contain_keys = []
        super().__init__(class_labels, cfg, must_contain_keys)
        self.model = self._init_model(cfg['model_type'])
        self.early_stopping = self.cfg['early_stopping']

    def _init_model(self, model_type):
        if model_type == 'xgboost':
            return XGBoost(self.class_labels, self.cfg)
        elif model_type == 'sgdc':
            return SGDC(self.class_labels, self.cfg)
        elif model_type == 'svm':
            return SVM(self.class_labels, self.cfg)
        elif model_type == 'rf':
            return RandomForest(self.class_labels, self.cfg)
        elif model_type == 'knn':
            return KNN(self.class_labels, self.cfg)
        elif model_type == 'nb':
            return NaiveBayes(self.class_labels, self.cfg)
        elif model_type == 'catboost':
            return CatBoost(self.class_labels, self.cfg)
        elif model_type == 'lightgbm':
            return LightGBM(self.class_labels, self.cfg)
        else:
            raise NotImplementedError('Model type: cnn|xgboost|knn|catboost|sgdc|svm are supported.')

    def _fit_regress(self, inputs, labels, eval_inputs=None, eval_labels=None):
        if phase == 'train':  # train時はパラメータ更新&trainのlossを算出
            if self.early_stopping:
                loss = self.model.fit(inputs, labels, eval_inputs, eval_labels)
            else:
                loss = self.model.fit(inputs, labels)
            self.fitted = self.model.fitted

        preds = self.model.predict(eval_inputs)

        if phase == 'val':  # validation時はlossのみ算出
            loss = self.criterion(torch.from_numpy(preds).float(), labels.float()).item()

        return loss, preds

    def _fit_classify(self, inputs, labels, eval_inputs=None, eval_labels=None):
        if self.early_stopping and isinstance(eval_inputs, np.ndarray):
            loss = self.model.fit(inputs, labels, eval_inputs, eval_labels)
        else:
            loss = self.model.fit(inputs, labels)
        self.fitted = self.model.fitted

        return loss

    def fit(self, inputs, labels, eval_inputs=None, eval_labels=None):
        if self.cfg['task_type'] == 'classify':
            return self._fit_classify(inputs, labels, eval_inputs=eval_inputs, eval_labels=eval_labels)
        else:
            return self._fit_regress(inputs, labels, eval_inputs=eval_inputs, eval_labels=eval_labels)

    def predict(self, inputs):
        return self.model.predict(inputs)
