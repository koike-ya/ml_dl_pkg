import logging
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore

from example_face import main
from ml.utils.config import ExptConfig, SGDConfig, AdamConfig

cs = ConfigStore.instance()
cs.store(name="config", node=ExptConfig)
cs.store(group="optim", name="sgd", node=SGDConfig)
cs.store(group="optim", name="adam", node=AdamConfig)


def experiment(cfg):
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("[%(name)s] [%(levelname)s] %(message)s"))
    console.setLevel(logging.INFO)
    logging.getLogger("ml").addHandler(console)

    cfg.expt_id = f"{cfg['model_type']}_{cfg['transform']}"
    expt_dir = Path(__file__).resolve().parent / 'output' / 'example_esc' / f"{cfg['expt_id']}"
    expt_dir.mkdir(exist_ok=True, parents=True)
    hyperparameters = {
        'lr': [1e-5],
    }
    cfg = dict(cfg)
    main(cfg, expt_dir, hyperparameters)


@hydra.main(config_name="config")
def hydra_main(cfg: ExptConfig):
    experiment(cfg=cfg)


if __name__ == '__main__':
    hydra_main()