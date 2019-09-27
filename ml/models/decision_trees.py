import catboost as ctb
import numpy as np
import xgboost as xgb

from ml.models.toolbox import BaseMLPredictor


def decision_trees_args(parser):

    decision_trees_parser = parser.add_argument_group("Decision tree-like model hyper parameters")
    decision_trees_parser.add_argument('--n-estimators', type=int, default=200)
    decision_trees_parser.add_argument('--max-depth', type=int, default=5)
    decision_trees_parser.add_argument('--reg-alpha', type=float, default=1.0, help='L1 regularization term on weights')
    decision_trees_parser.add_argument('--reg-lambda', type=float, default=1.0, help='L2 regularization term on weights')
    decision_trees_parser.add_argument('--subsample', type=float, default=0.8, help='Sample rate for bagging')

    return parser


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
            params['objective'] = 'reg:squarederror'
            self.model = xgb.XGBRegressor(**params)
        super(XGBoost, self).__init__(class_labels, cfg)

    def partial_fit(self, x, y):
        eval_metric = 'mlogloss' if self.classify else 'rmse'
        self.model.fit(x, y, eval_set=[(x, y)], eval_metric=eval_metric, verbose=False)
        return np.array(self.model.evals_result()['validation_0'][eval_metric]).mean()

    def predict(self, x):
        return self.model.predict(x)


class CatBoost(BaseMLPredictor):
    def __init__(self, class_labels, cfg):
        # TODO visualizationも試す
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
        if cfg['task_type'] == 'classify':
            params['eval_metric'] = 'Accuracy'
            self.model = ctb.CatBoostClassifier(**params)
        else:
            self.model = ctb.CatBoostRegressor(**params)
        super(CatBoost, self).__init__(class_labels, cfg)

    def partial_fit(self, x, y):
        self.model.fit(x, y, verbose=False)
        return self.model.best_score_['learn']['Accuracy']

    def predict(self, x):
        return self.model.predict(x).reshape((-1,))
