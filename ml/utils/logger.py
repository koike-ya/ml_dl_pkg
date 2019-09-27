import os

from tensorboardX import SummaryWriter


def to_np(x):
    return x.cpu().numpy()


class TensorBoardLogger(object):
    def __init__(self, id, log_dir, log_params):
        os.makedirs(log_dir, exist_ok=True)
        self.id = id
        self.tensorboard_writer = SummaryWriter(log_dir)
        self.log_params = log_params

    def update(self, epoch, values, parameters=None):
        # values = {
        #     'Avg Train Loss': values["loss"],
        #     'Avg recall 0': values["rec_0"],
        #     'Avg recall 1': values["rec_1"],
        #     'Avg recall AUC': values["auc"]
        # }
        self.tensorboard_writer.add_scalars(self.id, values, epoch + 1)
        # if self.log_params:
        #     for tag, value in parameters():
        #         tag = tag.replace('.', '/')
        #         self.tensorboard_writer.add_histogram(tag, to_np(value), epoch + 1)
        #         self.tensorboard_writer.add_histogram(tag + '/grad', to_np(value.grad), epoch + 1)

    def load_previous_values(self, start_epoch, values):
        raise NotImplementedError
        # loss_results = values["loss_results"][:start_epoch]
        # wer_results = values["wer_results"][:start_epoch]
        # cer_results = values["cer_results"][:start_epoch]
        #
        # for i in range(start_epoch):
        #     values = {
        #         'Avg Train Loss': loss_results[i],
        #         'Avg WER': wer_results[i],
        #         'Avg CER': cer_results[i]
        #     }
        #     self.tensorboard_writer.add_scalars(self.id, values, i + 1)
