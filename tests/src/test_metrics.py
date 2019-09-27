import unittest

from ml.src.metrics import Metric


class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.class_names = [0, 1, 2]
        self.metrics = [
            Metric('loss', direction='minimize', save_model=True),
            Metric('accuracy', direction='maximize'),
        ]

    def tearDown(self):
        pass

    def test___init__(self):
        metric_pattern = ['loss', 'recall', 'far', 'accuracy', 'confusion_matrix']

        test_pattern = []
        for metric_name in metric_pattern:
            test_pattern.append({
                'description': f'{metric_name}の場合の初期化',
                'metric': metric_name,
                'expected': metric_name
            })
        for test_case in test_pattern:
            actual = Metric(test_case['metric'], direction='minimize')
            with self.subTest(test_case['description']):
                self.assertEqual(test_case['expected'], actual.name)

    def test_update(self):
        # TODO 実装
        pass


if __name__ == '__main__':
    unittest.main()
