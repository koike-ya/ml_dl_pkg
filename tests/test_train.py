from __future__ import print_function, division

import argparse
from pathlib import Path
from unittest import TestCase

from wrapper.src.metrics import Metric
from wrapper.src.train import train
from wrapper.utils import train_args


class TestTrain(TestCase):

    def setUp(self):
        self.args = train_args().parse_args()
        self.args.data_path = Path(__file__).parent / '03_EURUSD_D.csv'
        self.class_names = ['bull', 'bear']

        self.metrics = [
            Metric('loss', initial_value=10000, inequality='less', save_model=True),
            Metric('recall', initial_value=0, inequality='more'),
            Metric('far', initial_value=0, inequality='more')]

    def tearDown(self):
        pass

    def test_train_model(self):
        pass

    def test_save_model(self):
        pass

    def test_update_by_epoch(self):
        pass

    def test_record_log(self):
        pass

    def test_train(self):
        train(self.args, self.class_names, lambda x: x, self.metrics)
