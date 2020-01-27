import numpy as np
import torch
import copy
from typing import Tuple
from apex import amp
from sklearn.exceptions import NotFittedError

from ml.models.base_model import BaseModel
from ml.models.rnn import construct_rnn, construct_cnn_rnn
from ml.models.cnn import construct_cnn, construct_logmel_cnn
from ml.models.ml_model import MLModel
from ml.models.pretrained_models import construct_pretrained, supported_pretrained_models


supported_nn_models = ['cnn', 'rnn', 'cnn_rnn', 'logmel_cnnrnn']


class NNModel(BaseModel):
    def __init__(self, class_labels, cfg):
        must_contain_keys = ['lr', 'weight_decay', 'momentum', 'learning_anneal']
        super().__init__(class_labels, cfg, must_contain_keys)
        self.device = torch.device('cuda' if cfg['cuda'] else 'cpu')
        self.feature_extract = cfg.get('feature_extract', False)
        self.model = self._init_model(transfer=cfg['transfer']).to(self.device)
        self.criterion = self.criterion.to(self.device)
        self.optimizer = self._set_optimizer()
        self.fitted = False
        self.amp = cfg.get('amp', False)
        if self.amp:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer)
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

    def _init_model(self, transfer=False):
        if transfer:
            orig_classes = self.class_labels
            self.class_labels = self.cfg['prev_classes']

        if self.cfg['model_type'] in supported_pretrained_models.keys():
            model = construct_pretrained(self.cfg, len(self.class_labels))
        elif self.cfg['model_type'] == 'rnn':
            model = construct_rnn(self.cfg, len(self.class_labels))
        elif self.cfg['model_type'] == 'cnn_rnn':
            n_dim = len(self.cfg['cnn_kernel_sizes'][0])
            model = construct_cnn_rnn(self.cfg, construct_cnn, len(self.class_labels), self.device, n_dim=n_dim)
        elif self.cfg['model_type'] == 'cnn':
            model = construct_cnn(self.cfg, use_as_extractor=False)
        elif self.cfg['model_type'] == 'logmel_cnnrnn':
            self.cfg['n_channels'] = 23
            self.cfg['image_size'] = (61, 129)
            n_dim = len(self.cfg['cnn_kernel_sizes'][0])
            model = construct_cnn_rnn(self.cfg, construct_cnn, len(self.class_labels), self.device, n_dim=n_dim)
        else:
            raise NotImplementedError(f"model_type should be in {supported_nn_models}, but {self.cfg['model_type']} selected.")

        if self.feature_extract:
            self.predictor = MLModel(self.class_labels, self.cfg, 'svm')

        if transfer:
            self.class_labels = orig_classes
            self.load_model(model)
            model.change_last_layer(len(orig_classes))

        print(f'Model Parameters: {get_param_size(model)}')

        return model

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

            if hasattr(self, 'predictor'):
                extracted_features = outputs
                if not self.predictor.fitted:
                    self.predictor.fit(copy.copy(extracted_features).detach(), labels, phase='train')
                outputs = torch.from_numpy(self.predictor.predict(copy.copy(extracted_features).detach()))

                pred_onehot = torch.zeros(outputs.size(0), len(self.class_labels))
                pred_onehot = pred_onehot.scatter_(1, outputs.view(-1, 1).type(torch.LongTensor), 1)
                outputs = pred_onehot.to(self.device)

            y_onehot = torch.zeros(labels.size(0), len(self.class_labels))
            y_onehot = y_onehot.scatter_(1, labels.view(-1, 1).type(torch.LongTensor), 1)

            loss = self.criterion(outputs, y_onehot.to(self.device))

            if hasattr(self, 'predictor'):
                loss = extracted_features.sum().to(self.device) - extracted_features.sum().to(self.device)
                loss += self.criterion(outputs, y_onehot.to(self.device))

            if phase == 'train':
                self.optimizer.zero_grad()
                if self.amp:
                    with amp.scale_loss(loss, self.optimizer) as loss:
                        loss.backward()
                else:
                    loss.backward(retain_graph=True)
                self.optimizer.step()

                if hasattr(self, 'predictor'):
                    self.predictor.fit(copy.copy(extracted_features).detach(), labels, phase=phase)

            _, preds = torch.max(outputs, 1)

        return loss.item(), preds.cpu().numpy()

    def _fit_regress(self, inputs, labels, phase) -> Tuple[float, np.ndarray]:
        with torch.set_grad_enabled(phase == 'train'):
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
        if self.cfg['task_type'] == 'classify':
            return self._fit_classify(inputs, labels, phase)
        else:
            return self._fit_regress(inputs, labels, phase)

    def save_model(self):
        torch.save(self.model.state_dict(), self.cfg['model_path'])

    def load_model(self, model=None):
        if model:
            self.model = model

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
            self.model.eval()
            preds = self.model(inputs)

            if self.cfg['task_type'] == 'classify':
                if hasattr(self, 'predictor'):
                    # TODO classifierも別ファイルに重みを保存しておいて、model_managerで読み込み
                    preds = torch.from_numpy(self.predictor.predict(preds.detach()))
                else:
                    _, preds = torch.max(preds, 1)

        return preds.cpu().numpy()

    def update_by_epoch(self, phase):
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] / self.cfg['learning_anneal']
        print(f"Learning rate annealed to: {g['lr']:.6f}")


def get_param_size(model):
    params = 0
    for p in model.parameters():
        tmp = 1
        for x in p.size():
            tmp *= x
        params += tmp
    return params