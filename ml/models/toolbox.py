import pickle

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


class BaseMLPredictor:
    def __init__(self, class_labels, cfg):
        self.class_labels = class_labels
        self.cfg = cfg
        self.fitted = False

    def save_model(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self, fname):
        with open(fname, 'rb') as f:
            self.model = pickle.load(f)
        self.fitted = True

    def partial_fit(self, x, y) -> np.float:
        self.fitted = True
        # lossを返却
        self.model.partial_fit(x, y, self.class_labels)
        return log_loss(y, self.model.predict_proba(x), labels=self.class_labels)

    def predict(self, x):
        if not self.fitted:
            raise NotFittedError(f'This MLModel instance is not fitted yet.')
        return self.model.predict(x)

    def predict_proba(self, x) -> np.float32:
        return self.model.predict_proba(x).astype(np.float32)


class KNN(BaseMLPredictor):
    def __init__(self, class_labels, cfg, dataloaders):
        super(KNN, self).__init__(class_labels, cfg)
        self.model = KNeighborsClassifier(n_neighbors=len(self.class_labels), n_jobs=-1)
        self.dataloaders = dataloaders
        self.x = self.dataloaders['train'].dataset.x.copy()
        self.y = self.dataloaders['train'].dataset.y.copy()
        self.model.fit(self.x, self.y)

    def partial_fit(self, x, y):
        # TODO 要修正
        return np.float(0.0)

    def predict(self, x):
        # if self.x.shape[0] != 1:
        # else:
            # raise NotFittedError(f'This MLModel instance(K-Nearest Neighbor) is not fitted yet.')
        return self.model.predict(x)

    def predict_proba(self, x):
        # knnは確率にできないためonehotにして対応する
        if self.x.shape[0] != 1:
            self.model.fit(self.x, self.y)
            return self.model.predict_proba(x).astype(np.float32)
        else:
            raise NotFittedError(f'This MLModel instance(K-Nearest Neighbor) is not fitted yet.')


class SGDC(BaseMLPredictor):
    def __init__(self, class_labels, cfg):
        class_weight = dict(zip(class_labels, cfg['loss_weight']))
        self.model = SGDClassifier(loss='log', alpha=cfg['lr'], shuffle=False, n_jobs=cfg['n_jobs'],
                                   random_state=cfg['seed'], learning_rate='optimal', class_weight=class_weight)
        super(SGDC, self).__init__(class_labels, cfg)


class SVM(BaseMLPredictor):
    def __init__(self, class_labels, cfg):
        class_weight = dict(zip(class_labels, cfg['loss_weight']))
        self.model = SVC(random_state=cfg['seed'])
        super(SVM, self).__init__(class_labels, cfg)

    def partial_fit(self, x, y) -> np.float:
        self.fitted = True
        # lossを返却
        self.model.fit(x, y)
        return log_loss(y, self.model.predict_proba(x), labels=self.class_labels)