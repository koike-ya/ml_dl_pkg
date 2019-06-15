from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
import numpy as np
import pickle


class BaseClassifier:
    def __init__(self, class_labels):
        self.class_labels = class_labels

    def save_model(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model_(self, fname):
        with open(fname, 'rb') as f:
            self.model = pickle.load(f)

    def partial_fit(self, x, y):
        self.model.partial_fit(x, y, self.class_labels)

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        return self.model.predict_proba(x)


class KNN(BaseClassifier):
    def __init__(self, class_labels):
        super(KNN, self).__init__(class_labels)
        self.model = KNeighborsClassifier(n_neighbors=len(self.class_labels))
        self.x = np.empty(1)
        self.y = np.empty(1)

    def partial_fit(self, x, y):
        if self.x.size == 1:
            self.x, self.y = x, y
        else:
            self.x = np.vstack((self.x, x))
            self.y = np.hstack((self.y, y))

    def predict(self, x):
        if self.x.size != 1:
            self.model.fit(self.x, self.y)
        return self.model.predict(x)


class SGDC(BaseClassifier):
    def __init__(self, class_labels):
        self.model = SGDClassifier()
        super(SGDC, self).__init__(class_labels)


class XGBoost(BaseClassifier):
    def __init__(self, class_labels):
        params = dict(
            learning_rate=0.1,
            n_estimators=100,
            max_depth=5,
            min_child_weight=1,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8,
            nthread=4,
            scale_pos_weight=1,
            seed=27,
            num_class=len(class_labels),
            missing=None, random_state=0, n_jobs=4, objective='multi:softmax')
        self.model = xgb.XGBClassifier(**params)
        super(XGBoost, self).__init__(class_labels)

    def partial_fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        self.model._le = LabelEncoder().fit(self.class_labels)
        return self.model.predict(x)


class LightGBM(BaseClassifier):
    def __call__(self):
        return lgb


def model_baseline(models, x, phase, batch=True):

    if not models:
        models = {'knn': MiniBatchKMeans(), 'sgdc': SGDC(), 'xgb': XGBoost()}

    preds = dict(models)

    for model_name, model in models.items():
        if phase == 'train':
            model.fit(x, y)
        preds[model_name] = model.predict(x)
    return models, preds
