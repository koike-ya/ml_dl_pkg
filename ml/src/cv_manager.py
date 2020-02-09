from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold


SUPPORTED_CV = {'k_fold': KFold, 'group': GroupKFold, 'stratified': StratifiedKFold}


class KFoldManager:
    """
    Manager for cross sklearn validation classes

    """
    def __init__(self, cv_name: str, n_splits: int):
        assert cv_name in SUPPORTED_CV.keys(), \
            f'Unsupported cv was selected. Supported cv is in {SUPPORTED_CV}. You specified {cv_name}'
        self.cv = SUPPORTED_CV[cv_name](n_splits=n_splits)

    def split(self, X, y, groups=None):
        return self.cv.split(X, y, groups)