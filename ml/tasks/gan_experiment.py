import logging
from copy import deepcopy

from ml.models.train_managers.acgan_train_manager import ACGANTrainManager
from ml.preprocess.preprocessor import Preprocessor
from ml.src.dataloader import set_dataloader

logger = logging.getLogger(__name__)


class GANExperimentor:
    def __init__(self, cfg, load_func, label_func, process_func=None, dataset_cls=None):
        self.cfg = cfg
        self.load_func = load_func
        self.label_func = label_func
        self.dataset_cls = dataset_cls
        self.train_manager = None
        self.process_func = process_func

    def _experiment(self, metrics, phases):
        dataloaders = {}
        for phase in phases:
            if not self.process_func:
                self.process_func = Preprocessor(self.cfg, phase).preprocess
            dataset = self.dataset_cls(self.cfg[f'{phase}_path'], self.cfg, phase, self.load_func, self.process_func,
                                       self.label_func)
            dataloaders[phase] = set_dataloader(dataset, phase, self.cfg)

        self.train_manager = ACGANTrainManager(self.cfg['class_names'], self.cfg, dataloaders, deepcopy(metrics))

        self.train_manager.train()
