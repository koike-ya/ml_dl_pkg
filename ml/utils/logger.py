from pathlib import Path

from tensorboardX import SummaryWriter


class TensorBoardLogger(object):
    def __init__(self, id, log_dir):
        Path(log_dir).mkdir(exist_ok=True, parents=True)
        self.id = id
        self.tensorboard_writer = SummaryWriter(log_dir)

    def update(self, epoch, values):
        self.tensorboard_writer.add_scalars(self.id, values, epoch + 1)

    def close(self):
        self.tensorboard_writer.close()