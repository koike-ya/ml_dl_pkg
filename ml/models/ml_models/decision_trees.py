import catboost as ctb
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from ml.models.ml_models.toolbox import BaseMLPredictor


def decision_trees_args(parser):

    decision_trees_parser = parser.add_argument_group("Decision tree-like model hyper parameters")
    decision_trees_parser.add_argument('--n-estimators', type=int, default=200)
    decision_trees_parser.add_argument('--n-leaves', type=int, default=32)
    decision_trees_parser.add_argument('--max-depth', type=int, default=5)
    decision_trees_parser.add_argument('--reg-alpha', type=float, default=1.0, help='L1 regularization term on weights')
    decision_trees_parser.add_argument('--reg-lambda', type=float, default=1.0, help='L2 regularization term on weights')
    decision_trees_parser.add_argument('--subsample', type=float, default=0.8, help='Sample rate for bagging')

    return parser


def get_feature_importance(model_cls, features):
    feature_importances = pd.DataFrame()
    feature_importances['feature'] = features
    feature_importances['importance'] = model_cls.model.feature_importances_
    feature_importances = feature_importances.sort_values(by='importance', ascending=False)
    return feature_importances


class XGBoost(BaseMLPredictor):
    def __init__(self, class_labels, cfg):
        self.classify = cfg['task_type'] == 'classify'
        params = dict(
            learning_rate=cfg['lr'],
            n_estimators=cfg['n_estimators'],
            max_depth=cfg['max_depth'],
            min_child_weight=1,
            min_split_loss=0,
            subsample=cfg['subsample'],
            colsample_bytree=0.8,
            n_jobs=cfg['n_jobs'],
            reg_lambda=cfg['reg_lambda'],
            reg_alpha=cfg['reg_alpha'],
            missing=None,
            random_state=cfg['seed'],
        )
        if cfg['task_type'] == 'classify':
            params['num_class'] = len(class_labels)
            params['objective'] = 'multi:softprob'
            self.model = xgb.XGBClassifier(**params)
        else:
            params['objective'] = 'reg:linear'
            self.model = xgb.XGBRegressor(**params)
        super(XGBoost, self).__init__(class_labels, cfg)

    def fit(self, x, y):
        eval_metric = 'mlogloss' if self.classify else 'rmse'
        self.model.fit(x, y, eval_set=[(x, y)], eval_metric=eval_metric, verbose=False)
        return np.array(self.model.evals_result()['validation_0'][eval_metric]).mean()

    def predict(self, x):
        return self.model.predict(x)


class CatBoost(BaseMLPredictor):
    def __init__(self, class_labels, cfg):
        # TODO visualizationも試す
        self.classify = cfg['task_type'] == 'classify'

        params = dict(
            iterations=cfg['n_estimators'],
            depth=cfg['max_depth'],
            learning_rate=cfg['lr'],
            random_seed=cfg['seed'],
            has_time=True,
            reg_lambda=cfg['reg_lambda'],
            class_weights=cfg['loss_weight'],
            bootstrap_type='Bernoulli',
            subsample=cfg['subsample'],
            task_type='GPU' if cfg['cuda'] else 'CPU',
        )
        if self.classify:
            params['eval_metric'] = 'Accuracy'
            self.model = ctb.CatBoostClassifier(**params)
        else:
            del params['class_weights']
            self.model = ctb.CatBoostRegressor(**params)
        super(CatBoost, self).__init__(class_labels, cfg)

    def fit(self, x, y):
        self.model.fit(x, y, verbose=False)
        if self.classify:
            return self.model.best_score_['learn']['Accuracy']
        else:
            return self.model.best_score_['learn']['RMSE']

    def predict(self, x):
        return self.model.predict(x).reshape((-1,))


class LightGBM(BaseMLPredictor):
    def __init__(self, class_labels, cfg):
        # TODO visualizationも試す
        self.classify = cfg['task_type'] == 'classify'

        self.params = dict(
            num_leaves=cfg['n_leaves'],
            learning_rate=cfg['lr'],
            max_depth=cfg['max_depth'],
            subsample=cfg['subsample'],
            colsample_bytree=0.8,
            n_jobs=cfg['n_jobs'],
            reg_lambda=cfg['reg_lambda'],
            reg_alpha=cfg['reg_alpha'],
            # class_weight={i: weight for i, weight in enumerate(cfg['loss_weight'])},
            class_weight='balanced',
            missing=None,
            random_state=cfg['seed'],
            max_bin=255,
            num_iterations=1000,
        )
        if self.classify:
            if len(cfg['class_names']) == 2:
                self.params['metric'] = 'binary_logloss'
                self.params['objective'] = 'binary'
            else:
                self.params['num_class'] = len(cfg['class_names'])
                self.params['metric'] = 'multi_logloss'
            self.model = lgb.LGBMClassifier(**self.params)
        else:
            self.params['objective'] = 'regression'
            self.params['metric'] = 'rmse'
            self.model = lgb.LGBMRegressor(**self.params, num_trees=cfg['n_estimators'])
        super(LightGBM, self).__init__(class_labels, cfg)

    def fit(self, x, y, eval_x=None, eval_y=None):
        eval_set = [(x, y)]
        if isinstance(eval_x, np.ndarray):
            eval_set.append((eval_x, eval_y))

            # return self.model.best_score_['training'][self.params['metric']]
        self.model.fit(x, y, eval_set=eval_set, verbose=50, early_stopping_rounds=20)
        return list(self.model.best_score_.keys())[-1]

    def predict(self, x):
        return self.model.predict(x).reshape((-1,))
