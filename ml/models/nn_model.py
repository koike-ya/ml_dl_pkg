import numpy as np
import torch
from typing import Tuple
from sklearn.exceptions import NotFittedError

from ml.models.base_model import BaseModel
from ml.models.rnn import construct_rnn, construct_cnn_rnn
from ml.models.cnn import construct_cnn


supported_nn_models = ['cnn', 'rnn', 'cnn_rnn']


class NNModel(BaseModel):
    def __init__(self, class_labels, cfg):
        must_contain_keys = ['lr', 'weight_decay', 'momentum', 'learning_anneal']
        super().__init__(class_labels, cfg, must_contain_keys)
        self.device = torch.device('cuda' if cfg['cuda'] else 'cpu')
        self.model = self._init_model().to(self.device)
        self.criterion = self.criterion.to(self.device)
        self.optimizer = self._set_optimizer()
        self.fitted = False

    def _init_model(self):
        if self.cfg['model_type'] == 'rnn':
            return construct_rnn(self.cfg, len(self.class_labels))
        elif self.cfg['model_type'] == 'cnn_rnn':
            return construct_cnn_rnn(self.cfg, len(self.class_labels), self.device)
        elif self.cfg['model_type'] == 'cnn':
            return construct_cnn(self.cfg, use_as_extractor=False)
            # cnn_maker = CNNMaker(in_channels=self.n_channels, image_size=self.image_size, cfg=self.cfg,
            #                      n_classes=self.n_classes)
            # return cnn_maker.construct_cnn()
        else:
            raise NotImplementedError('model_type should be either rnn or cnn, nn would be implemented in the future.')

    def _set_optimizer(self):
        supported_optimizers = {
            'adam': torch.optim.Adam(self.model.parameters(), lr=self.cfg['lr'],
                                     weight_decay=self.cfg['weight_decay']),
            'rmsprop': torch.optim.RMSprop(self.model.parameters(), lr=self.cfg['lr'],
                                           weight_decay=self.cfg['weight_decay'], momentum=self.cfg['momentum']),
            'sgd': torch.optim.SGD(self.model.parameters(), lr=self.cfg['lr'], weight_decay=self.cfg['weight_decay'],
                                   momentum=self.cfg['momentum'], nesterov=True)
        }
        return supported_optimizers[self.cfg['optimizer']]

    def _fit_classify(self, inputs, labels, phase) -> Tuple[float, np.ndarray]:
        with torch.set_grad_enabled(phase == 'train'):
            outputs = self.model(inputs)
            y_onehot = torch.zeros(labels.size(0), len(self.class_labels))
            y_onehot = y_onehot.scatter_(1, labels.view(-1, 1).type(torch.LongTensor), 1)
            loss = self.criterion(outputs, y_onehot.to(self.device))

            if phase == 'train':
                loss.backward(retain_graph=True)
                self.optimizer.step()

            _, preds = torch.max(outputs, 1)

        return loss.item(), preds.cpu().numpy()

    def _fit_regress(self, inputs, labels, phase) -> Tuple[float, np.ndarray]:
        with torch.set_grad_enabled(phase == 'train'):
            preds = self.model(inputs)
            loss = self.criterion(preds, labels.float())

            if phase == 'train':
                loss.backward(retain_graph=True)
                self.optimizer.step()

        return loss.item(), preds.cpu().detach().numpy()

    def anneal_lr(self, learning_anneal):
        param_groups = self.optimizer.param_groups
        for g in param_groups:
            g['lr'] = g['lr'] / learning_anneal
        print('Learning rate annealed to: {lr:.6f}'.format(lr=g['lr']))

    def fit(self, inputs, labels, phase) -> Tuple[float, np.ndarray]:
        self.fitted = True
        self.optimizer.zero_grad()
        if self.cfg['task_type'] == 'classify':
            return self._fit_classify(inputs, labels, phase)
        else:
            return self._fit_regress(inputs, labels, phase)

    def save_model(self):
        torch.save(self.model.state_dict(), self.cfg['model_path'])

    def load_model(self):
        try:
            self.model.load_state_dict(torch.load(self.cfg['model_path'], map_location=self.device))
            self.model.to(self.device)
            print('Saved model loaded.')
        except FileNotFoundError as e:
            print(e)
            print(f"trained model file doesn't exist at {self.cfg['model_path']}")
            exit(1)

        self.fitted = True

    def predict(self, inputs):  # NNModelは自身がfittedを管理している
        if not self.fitted:
            raise NotFittedError(f'This NNModel instance is not fitted yet.')

        with torch.set_grad_enabled(False):
            preds = self.model(inputs)
            if self.cfg['task_type'] == 'classify':
                _, preds = torch.max(preds, 1)

        return preds.cpu().numpy()

    def update_by_epoch(self, phase):
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] / self.cfg['learning_anneal']
        print(f"Learning rate annealed to: {g['lr']:.6f}")
