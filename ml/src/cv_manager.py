from enum import Enum

import pandas as pd
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold

CV_KIND = {'k_fold': KFold, 'group': GroupKFold, 'stratified': StratifiedKFold}


class SupportedCV(Enum):
    none = ''
    k_fold = 'k_fold'
    group = 'group'
    stratified = 'stratified'


class KFoldManager:
    """
    Manager for cross sklearn validation classes

    """
    def __init__(self, cv_name: str, n_splits: int, groups=None):
        self.cv = GroupKFold(n_splits=n_splits) if isinstance(groups, pd.Series) else CV_KIND[cv_name](n_splits=n_splits)
        self.groups = groups

    def split(self, X, y):
        return self.cv.split(X, y, self.groups)
#
#
# class NewKFoldManager:
#     """
#     Manager for cross sklearn validation classes
#
#     """
#     def __init__(self, cv_name: str, n_splits: int):
#         self.cv = CV_KIND[cv_name](n_splits=n_splits)
#
#     def split(self, X, y, groups=None):
#         return self.cv.split(X, y, groups)
