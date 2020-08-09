from enum import Enum

from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold

SUPPORTED_CV = {'k_fold': KFold, 'group': GroupKFold, 'stratified': StratifiedKFold}


class SupportedCV(Enum):
    none = ''
    k_fold = 'k_fold'
    group = 'group'
    stratified = 'stratified'


class KFoldManager:
    """
    Manager for cross sklearn validation classes

    """
    def __init__(self, cv_name: str, n_splits: int):
        self.cv = SUPPORTED_CV[cv_name](n_splits=n_splits)

    def split(self, X, y, groups=None):
        return self.cv.split(X, y, groups)


class NewKFoldManager:
    """
    Manager for cross sklearn validation classes

    """
    def __init__(self, cv_name: str, n_splits: int):
        self.cv = SUPPORTED_CV[cv_name](n_splits=n_splits)

    def split(self, X, y, groups=None):
        return self.cv.split(X, y, groups)
