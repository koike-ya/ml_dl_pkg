from pathlib import Path
import numpy as np
import pandas as pd
from ml.models.nn_models.cnn_1d import construct_1dcnn
from ml.models.model_managers.nn_model_manager import NNModelManager
import torch
import json
import matplotlib.pyplot as plt


class DummyClass:
    def __init__(self, cfg):
        self.model = construct_1dcnn(cfg)

    def load_model(self, model_path, device):
        self.model.load_state_dict(torch.load(model_path, map_location=device))


if __name__ == '__main__':
    expt_dir = Path('/home/tomoya/workspace/research/covid19/ml_pkg/output/example_esc/1dcnn_None')
    # with open(expt_dir / 'best_parameters.txt', 'r') as f:
    #     best_parameters = json.load(f)
    #
    # best_parameters = {key: str(value) for key, value in best_parameters.items()}

    parameters = "[32, 64, 'M']_[[1, 64], [1, 16], [1, 64]]_[[1, 2], [1, 2], [1, 64]]_None_0.0001_1dcnn"

    # with open(expt_dir / f"{'_'.join(best_parameters.values())}.txt", 'r') as f:
    with open(expt_dir / f"{parameters}.txt", 'r') as f:
        cfg = json.load(f)
    for key in cfg.keys():
        if cfg[key].startswith('['):
            cfg[key] = json.loads(cfg[key])

    cfg.update(
        {
            'cnn_channel_list': [32, 64, 'M'],
            'cnn_kernel_sizes': [[1, 64], [1, 16], [1, 64]],
            'cnn_stride_sizes': [[1, 2], [1, 2], [1, 64]],
            'transform': None,
            'lr': 1e-4,
        }
    )

    device = torch.device('cuda' if cfg['cuda'] else 'cpu')

    # dummy_nn_model_manager = DummyClass(cfg)
    # dummy_nn_model_manager.load_model(expt_dir / f"{parameters}.pth", device)

    cfg['transfer'] = False
    cfg['lr'] = float(cfg['lr'])
    cfg['weight_decay'] = float(cfg['weight_decay'])
    cfg['momentum'] = float(cfg['momentum'])
    nn_model_manager = NNModelManager(list(range(10)), cfg)

    x = torch.from_numpy(np.sin(np.linspace(0, 2 * np.pi, 11025 * 5))).to(device)
    feature_map = nn_model_manager.model.get_1dconv_responce(x).cpu().detach().numpy()
    plt.imshow(feature_map)
    plt.savefig(expt_dir / 'frequency_responce.png')
