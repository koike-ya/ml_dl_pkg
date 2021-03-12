import logging
from typing import Tuple

import numpy as np
import torch
from apex import amp
from ml.models.model_managers.base_model_manager import BaseModelManager
from ml.models.nn_models.attention import AttentionClassifier
from ml.models.nn_models.cnn import construct_cnn
from ml.models.nn_models.cnn_rnn import construct_cnn_rnn, CNNRNNConfig
from ml.models.nn_models.cnn_1d import construct_1dcnn
from ml.models.nn_models.logmel_cnn import construct_logmel_cnn
from ml.models.nn_models.multitask_panns_model import construct_multitask_panns
from ml.models.nn_models.multitask_predictor import MultitaskPredictor
from ml.models.nn_models.nn import construct_nn
from ml.models.nn_models.nn_utils import get_param_size
from ml.models.nn_models.panns_cnn14 import construct_panns
from ml.models.nn_models.pretrained_models import construct_pretrained, supported_pretrained_models
from ml.models.nn_models.mil_wrapper import construct_mil
from ml.models.nn_models.rnn import construct_rnn, RNNConfig
from ml.utils.nn_config import SGDConfig, AdamConfig
from omegaconf import OmegaConf
from sklearn.exceptions import NotFittedError

logger = logging.getLogger(__name__)


class StackedNNModel(torch.nn.Module):
    def __init__(self, cfg, class_labels, multitask=False):
        super(StackedNNModel, self).__init__()
        self.cfg = cfg
        self.device = torch.device('cuda' if cfg.cuda else 'cpu')
        self.class_labels = class_labels
        self.feature_extractors = [
            construct_cnn(cfg=self.cfg).to(self.device),
            # construct_cnn_rnn(cfg=self.cfg, construct_cnn_func=construct_cnn, n_classes=len(class_labels)).to(self.device),
        ]
        # hidden_size = cfg.rnn_hidden_size * 2 if cfg.bidirectional else cfg.rnn_hidden_size
        if multitask:
            self.predictor = MultitaskPredictor(self.feature_extractors[-1].predictor.in_features,
                                                cfg.n_labels_in_each_task, self.device)
        elif cfg.attention:
            self.predictor = AttentionClassifier(len(class_labels), hidden_size, d_attn=cfg.d_attn, n_heads=cfg.n_heads).to(self.device)
        else:
            self.predictor = self.feature_extractors[-1].predictor.to(self.device)

    def forward(self, x):
        for feature_extractor in self.feature_extractors:
            x = feature_extractor.extract_feature(x)

        return self.predictor(x)


class NNModelManager(BaseModelManager):
    def __init__(self, class_labels, cfg):
        super().__init__(class_labels, cfg)
        self.device = torch.device('cuda' if cfg.cuda else 'cpu')
        self.model = self._instantiate_model(class_labels).to(self.device)
        self.mixup_alpha = cfg.mixup_alpha
        if self.mixup_alpha:
            self._orig_criterion = self.criterion.to(self.device)
            self.criterion = self._mixup_criterion(lamb=1.0)
        else:
            self.criterion = self.criterion.to(self.device)
        self.optimizer = self._set_optimizer()
        self.fitted = False
        self.amp = cfg.amp
        if self.amp:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer)
        if torch.cuda.device_count() > 1 and OmegaConf.get_type(cfg) not in [RNNConfig, CNNRNNConfig]:
            self.model = torch.nn.parallel.DataParallel(self.model)

    def _instantiate_model(self, class_labels):
        if self.cfg.model_name in supported_pretrained_models.keys():
            model = construct_pretrained(self.cfg, len(self.class_labels))
        elif self.cfg.model_name == 'nn':
            model = construct_nn(self.cfg)
        elif self.cfg.model_name == 'rnn':
            model = construct_rnn(self.cfg, len(self.class_labels))
        elif self.cfg.model_name == 'cnn_rnn':
            model = construct_cnn_rnn(self.cfg, construct_cnn, len(self.class_labels))
        elif self.cfg.model_name == 'cnn':
            model = construct_cnn(self.cfg, use_as_extractor=False)
        elif self.cfg.model_name == 'logmel_cnn':
            model = construct_logmel_cnn(self.cfg)
        elif self.cfg.model_name == 'panns':
            model = construct_panns(self.cfg)
        elif self.cfg.model_name == 'multitask_panns':
            model = construct_multitask_panns(self.cfg)
        elif self.cfg.model_name == '1dcnn':
            model = construct_1dcnn(self.cfg)
        else:
            raise NotImplementedError(
                'model_type should be either rnn or cnn, nn would be implemented in the future.')

        if self.cfg.train_mil:
            model = construct_mil(model, self.cfg)

        # model = StackedNNModel(self.cfg, class_labels, multitask=False)
        logger.info(f'Model Parameters: {get_param_size(model)}')

        return model

    def _mixup_criterion(self, lamb):
        def mixup_criterion_func(pred, lables):
            batch_size = pred.size(0)
            y_orig, y_shuffled = lables[:batch_size], lables[batch_size:]
            return lamb * self._orig_criterion(pred, y_orig) + (1 - lamb) * self._orig_criterion(pred, y_shuffled)

        if lamb < 1:
            return mixup_criterion_func
        else:
            return self._orig_criterion

    def _set_optimizer(self):
        if OmegaConf.get_type(self.cfg.optim) == AdamConfig:
            return torch.optim.Adam(self.model.parameters(), lr=self.cfg.optim.lr,
                                    weight_decay=self.cfg.optim.weight_decay)
        elif OmegaConf.get_type(self.cfg.optim) == SGDConfig:
            return torch.optim.SGD(self.model.parameters(), lr=self.cfg.optim.lr, momentum=self.cfg.optim.momentum,
                                   weight_decay=self.cfg.optim.weight_decay, nesterov=True)

    def _mixup_data(self, inputs, labels, phase):
        # To be sure lamb is under 1.0 if phase == 'train' and equals 1.0 if phase != 'train'
        suffix = 0.00001
        if phase == 'train':
            lamb = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            index = torch.randperm(inputs.size(0)).to(self.device)
            inputs = lamb * inputs + (1 - lamb) * inputs[index, :]

            labels_orig, labels_shuffled = labels, labels[index]
            labels = torch.cat((labels_orig, labels_shuffled), dim=0)
        else:
            lamb = 1.0 + suffix

        return inputs, labels, lamb - suffix

    def _fit_classify(self, inputs, labels, phase) -> Tuple[float, np.ndarray]:
        if self.mixup_alpha:
            inputs, labels, lamb = self._mixup_data(inputs, labels, phase)
            self.criterion = self._mixup_criterion(lamb)

        with torch.set_grad_enabled(phase == 'train'):
            self.model.train() if phase == 'train' else self.model.eval()

            outputs = self.model(inputs)

            if labels.dim() == 1:   # Not softlabel
                labels = torch.eye(len(self.class_labels))[labels].to(self.device)

            loss = self.criterion(outputs, labels)

            if phase == 'train':
                self.optimizer.zero_grad()
                if self.amp:
                    with amp.scale_loss(loss, self.optimizer) as loss:
                        loss.backward()
                else:
                    loss.backward(retain_graph=True)

                if self.cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), *list(self.cfg.grad_clip))

                self.optimizer.step()

            if self.cfg.return_prob:
                preds = outputs.detach()
            else:
                _, preds = torch.max(outputs, 1)

        return loss.item(), preds.cpu().numpy()

    def _fit_regress(self, inputs, labels, phase) -> Tuple[float, np.ndarray]:
        if self.mixup_alpha:
            inputs, labels, lamb = self._mixup_data(inputs, labels, phase)
            self.criterion = self._mixup_criterion(lamb)

        with torch.set_grad_enabled(phase == 'train'):
            self.model.train() if phase == 'train' else self.model.eval()

            preds = self.model(inputs)

            if hasattr(self, 'predictor'):
                extracted_features = preds
                preds = self.predictor.predict(extracted_features)

            loss = self.criterion(preds, labels.float())

            if phase == 'train':
                self.optimizer.zero_grad()
                if self.amp:
                    with amp.scale_loss(loss, self.optimizer) as loss:
                        loss.backward()
                else:
                    loss.backward(retain_graph=True)

                if self.cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), *list(self.cfg.grad_clip))

                self.optimizer.step()

                if hasattr(self, 'predictor'):
                    self.predictor.fit(extracted_features, labels, phase=phase)

        return loss.item(), preds.cpu().detach().numpy()

    def anneal_lr(self, learning_anneal):
        param_groups = self.optimizer.param_groups
        for g in param_groups:
            g['lr'] = g['lr'] / learning_anneal

    def get_lr(self):
        return self.optimizer.param_groups[-1]['lr']

    def fit(self, inputs, labels, phase) -> Tuple[float, np.ndarray]:
        self.fitted = True
        self.optimizer.zero_grad()
        if self.cfg.task_type.value == 'classify':
            return self._fit_classify(inputs, labels, phase)
        else:
            return self._fit_regress(inputs, labels, phase)

    def save_model(self):
        torch.save(self.model.state_dict(), self.cfg.model_path)

    def load_model(self, model=None):
        if model:
            self.model = model

        try:
            self.model.load_state_dict(torch.load(self.cfg.model_path, map_location=self.device))
            self.model.to(self.device)
            logger.info('Saved model loaded.')
        except FileNotFoundError as e:
            logger.info(e)
            logger.info(f"trained model file doesn't exist at {self.cfg.model_path}")
            exit(1)

        self.fitted = True

    def predict(self, inputs):  # NNModelManagerは自身がfittedを管理している
        if not self.fitted:
            raise NotFittedError(f'This NNModelManager instance is not fitted yet.')

        with torch.set_grad_enabled(False):
            self.model.eval()
            preds = self.model(inputs)

            if self.cfg.task_type.value == 'classify':
                if hasattr(self, 'predictor'):
                    # TODO classifierも別ファイルに重みを保存しておいて、train_managerで読み込み
                    preds = torch.from_numpy(self.predictor.predict(preds.detach()))

                if not self.cfg.return_prob:
                    _, preds = torch.max(preds, 1)

        return preds.cpu().numpy()

    def update_by_epoch(self, phase):
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] / self.cfg.learning_anneal
        logger.info(f"Learning rate annealed to: {g['lr']:.6f}")
