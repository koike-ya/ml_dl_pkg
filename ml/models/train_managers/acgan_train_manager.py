import logging

logger = logging.getLogger(__name__)

from ml.models.train_managers.base_train_manager import BaseTrainManager
from ml.models.model_managers.acgan_model_manager import ACGANModelManager
import torch


class ACGANTrainManager(BaseTrainManager):
    def __init__(self, class_labels, cfg, dataloaders, metrics):
        cfg['n_classes'] = len(cfg['class_names'])
        cfg['channels'] = 1
        super().__init__(class_labels, cfg, dataloaders, metrics)

    def _init_model_manager(self):
        return ACGANModelManager(self.class_labels, self.cfg)

    def _init_device(self) -> torch.device:
        if self.cfg['cuda']:
            device = torch.device("cuda")
            torch.cuda.set_device(self.cfg['gpu_id'])
        else:
            device = torch.device("cpu")

        return device

    def train(self, model_manager=None, with_validate=True, only_validate=False):
        for epoch in range(self.cfg['gan_epochs']):
            for i, (imgs, labels) in enumerate(self.dataloaders['train']):

                batch_size = imgs.shape[0]

                # Adversarial ground truths
                valid = torch.FloatTensor(batch_size, 1).fill_(1.0).to(self.device)
                fake = torch.FloatTensor(batch_size, 1).fill_(0.0).to(self.device)

                # Configure input
                real_imgs = imgs.type(torch.FloatTensor).to(self.device)
                labels = labels.type(torch.LongTensor).to(self.device)

                g_loss, d_loss, d_acc = self.model_manager.train(valid, batch_size, fake, real_imgs, labels)

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]"
                    % (epoch, self.cfg['epochs'], i, len(self.dataloaders['train']), d_loss.item(), 100 * d_acc, g_loss.item())
                )
                batches_done = epoch * len(self.dataloaders['train']) + i
                if batches_done % self.cfg['sample_interval'] == 0:
                    self.model_manager.sample_image(n_row=10, batches_done=batches_done)
