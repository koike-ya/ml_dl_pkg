from __future__ import print_function, division

from pathlib import Path
import pandas as pd
from unittest import TestCase
import torch
import numpy as np

from wrapper.models.models import Models


class TestTrain(TestCase):

    def setUp(self):
        self.test_data = pd.read_csv(Path(__file__).parent / '03_EURUSD_D.csv', encoding='shift_jis')
        self.test_data = self.test_data.iloc[:, 2:]
        self.input_shape = self.test_data.shape
        self.class_names = ['bull', 'bear']
        # table data version
        self.table_models = Models(n_classes=len(self.class_names), batch_size=self.input_shape[0],
                                   height=self.input_shape[1])

    def tearDown(self):
        pass

    def test_cnn(self):
        test_pattern = [
            {
                'describe': 'channel数が1で画像サイズが200×4, 4層で256channelにするCNNを使用した場合',
                'in_channel': 1,
                'cfg': [(32, (2, 4), (1, 3), (1, 2)),
                        (64, (2, 4), (1, 3), (1, 1)),
                        (128, (2, 3), (1, 2), (1, 1)),
                        (256, (4, 4), (2, 3), (1, 2))],
                'expect_feature_extractors_output_size': 25856,
            },
            {
                'describe': 'channel数が10で画像サイズが200×4, 4層で256channelにするCNNを使用した場合',
                'in_channel': 10,
                'cfg': [(32, (2, 4), (1, 3), (1, 2)),
                        (64, (2, 4), (1, 3), (1, 1)),
                        (128, (2, 3), (1, 2), (1, 1)),
                        (256, (4, 4), (2, 3), (1, 2))],
                'expect_feature_extractors_output_size': 25856,
            },
        ]
        orig_images = np.array([self.test_data.values] * self.test_data.shape[0])
        for test_case in test_pattern:
            models = Models(n_classes=len(self.class_names), batch_size=self.input_shape[0],
                   height=self.input_shape[0], width=self.input_shape[1], n_channels=test_case['in_channel'])
            model = models.select_model(model_kind='cnn', cfg=test_case['cfg'])
            input_images = np.array([orig_images] * test_case['in_channel'])
            input_tensor = torch.Tensor(input_images).reshape(input_images.shape[1], test_case['in_channel'],
                                                             input_images.shape[2], input_images.shape[3])
            self.assertEqual(model.classifier[0].in_features, test_case['expect_feature_extractors_output_size'])

    # elif args.model_name == '2dcnn_2':
    #     model = cnn_16_751_751(n_labels=len(class_names))
    # elif args.model_name == 'rnn':
    #     cnn, out_ftrs = cnn_ftrs_16_751_751(eeg_conf)
    #     model = RNN(cnn, out_ftrs, args.batch_size, args.rnn_type, class_names, eeg_conf=eeg_conf,
    #                 rnn_hidden_size=args.rnn_hidden_size, nb_layers=args.rnn_n_layers)
    # elif args.model_name == '3dcnn':
    #     model = cnn_1_16_751_751(n_labels=len(class_names))
    # elif args.model_name == 'xgboost':
    #     model = XGBoost(list(range(len(class_names))))
    # elif args.model_name == 'sgdc':
    #     model = SGDC(list(range(len(class_names))))
    # elif args.model_name in ['kneighbor', 'knn']:
    #     args.model_name = 'kneighbor'
    #     model = KNN(list(range(len(class_names))))