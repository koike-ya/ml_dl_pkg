import numpy as np
import torch

from ml.models.base_model import BaseModel
from ml.models.decision_trees import CatBoost, XGBoost, LightGBM
from ml.models.toolbox import KNN, SGDC, SVM


class MLModel(BaseModel):
    def __init__(self, class_labels, cfg):
        must_contain_keys = []
        super().__init__(class_labels, cfg, must_contain_keys)
        self.model = self._init_model(cfg['model_type'])

    def _init_model(self, model_type):
        if model_type == 'xgboost':
            return XGBoost(self.class_labels, self.cfg)
        elif model_type == 'sgdc':
            return SGDC(self.class_labels, self.cfg)
        elif model_type == 'svm':
            return SVM(self.class_labels, self.cfg)
        elif model_type == 'knn':
            return KNN(self.class_labels, self.cfg)
        elif model_type == 'catboost':
            return CatBoost(self.class_labels, self.cfg)
        elif model_type == 'lightgbm':
            return LightGBM(self.class_labels, self.cfg)
        else:
            raise NotImplementedError('Model type: cnn|xgboost|knn|catboost|sgdc|svm are supported.')

    def _fit_regress(self, inputs, labels, phase):
        if phase == 'train':  # train時はパラメータ更新&trainのlossを算出
            loss = self.model.fit(inputs, labels.numpy())
            self.fitted = self.model.fitted

        preds = self.model.predict(inputs)

        if phase == 'val':  # validation時はlossのみ算出
            loss = self.criterion(torch.from_numpy(preds).float(), labels.float()).item()

        return loss, preds

    def _fit_classify(self, inputs, labels, phase):
        if phase == 'train':  # train時はパラメータ更新&trainのlossを算出
            loss = self.model.fit(inputs, labels.numpy())
            self.fitted = self.model.fitted

        outputs = self.model.predict_proba(inputs)
        preds = np.argmax(outputs, 1)

        if phase == 'val':  # validation時はlossのみ算出
            y_onehot = torch.zeros(labels.size(0), len(self.class_labels))
            y_onehot = y_onehot.scatter_(1, labels.view(-1, 1).type(torch.LongTensor), 1)
            loss = self.criterion(torch.from_numpy(outputs), y_onehot).item()

        return loss, preds

    def fit(self, inputs, labels, phase):
        inputs, labels = inputs.cpu().numpy(), labels.cpu()
        if self.cfg['task_type'] == 'classify':
            return self._fit_classify(inputs, labels, phase)
        else:
            return self._fit_regress(inputs, labels, phase)

    def predict(self, inputs):
        return self.model.predict(inputs.cpu().numpy())
