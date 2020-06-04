import logging

from apex import amp

logger = logging.getLogger(__name__)

from ml.models.model_managers.base_model_manager import BaseModelManager
from ml.models.nn_models.nn_utils import get_param_size
import numpy as np

import torch
from ml.models.nn_models.acgan import Generator, Discriminator
from torchvision.utils import save_image


class ACGANModelManager(BaseModelManager):
    def __init__(self, class_labels, cfg):
        must_contain_keys = ['lr', 'weight_decay', 'momentum', 'learning_anneal']
        super().__init__(class_labels, cfg, must_contain_keys)
        self.device = torch.device('cuda' if cfg['cuda'] else 'cpu')
        self.generator, self.discriminator = self._init_model()
        # Loss functions
        self.adversarial_loss = torch.nn.BCELoss().to(self.device)
        self.auxiliary_loss = torch.nn.CrossEntropyLoss().to(self.device)
        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(),
                                            lr=cfg['gan_lr'], betas=(self.cfg['b1'], self.cfg['b2']))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(),
                                            lr=cfg['gan_lr'], betas=(self.cfg['b1'], self.cfg['b2']))
        self.fitted = False
        self.amp = cfg.get('amp', False)
        if self.amp:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer)
        if torch.cuda.device_count() > 1:
            self.generator = torch.nn.DataParallel(self.generator)
            self.discriminator = torch.nn.DataParallel(self.discriminator)

    def _init_model(self):
        # Initialize generator and discriminator
        generator = Generator(self.cfg)
        discriminator = Discriminator(self.cfg)

        logger.info(f'Generator Parameters: {get_param_size(generator)}')
        logger.info(f'Discriminator Parameters: {get_param_size(discriminator)}')

        return generator.to(self.device), discriminator.to(self.device)

    def fit(self):
        pass

    def predict(self):
        pass

    def train(self, valid, batch_size, fake, real_imgs, labels):
        g_loss, gen_imgs, gen_labels = self.train_generator(valid, batch_size)
        d_real_loss, d_fake_loss, d_acc = self.train_discriminator(valid, fake, real_imgs, labels, gen_imgs, gen_labels)

        return g_loss, d_real_loss, d_fake_loss, d_acc

    def train_generator(self, valid, batch_size):
        self.optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = torch.FloatTensor(np.random.normal(0, 1, (batch_size, self.cfg['gan_latent_dim']))).to(self.device)
        gen_labels = torch.LongTensor(np.random.randint(0, self.cfg['n_classes'], batch_size)).to(self.device)

        # Generate a batch of images
        gen_imgs = self.generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity, pred_label = self.discriminator(gen_imgs)
        g_loss = 0.5 * (self.adversarial_loss(validity, valid) + self.auxiliary_loss(pred_label, gen_labels))
        g_loss *= self.cfg['gen_weight']
        g_loss.backward()
        self.optimizer_G.step()

        return g_loss, gen_imgs, gen_labels

    def train_discriminator(self, valid, fake, real_imgs, labels, gen_imgs, gen_labels):
        self.optimizer_D.zero_grad()

        # Loss for real images
        real_pred, real_aux = self.discriminator(real_imgs)
        d_real_loss = (self.adversarial_loss(real_pred, valid) + self.auxiliary_loss(real_aux, labels)) / 2

        # Loss for fake images
        fake_pred, fake_aux = self.discriminator(gen_imgs.detach())
        d_fake_loss = (self.adversarial_loss(fake_pred, fake) + self.auxiliary_loss(fake_aux, gen_labels)) / 2

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        # Calculate discriminator accuracy
        pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
        gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
        d_acc = np.mean(np.argmax(pred, axis=1) == gt)

        d_loss.backward()
        self.optimizer_D.step()

        return d_real_loss, d_fake_loss, d_acc

    def sample_image(self, n_row, batches_done):
        """Saves a grid of generated digits ranging from 0 to n_classes"""
        # Sample noise
        z = torch.FloatTensor(np.random.normal(0, 1, (n_row ** 2, self.cfg['gan_latent_dim']))).to(self.device)
        # Get labels ranging from 0 to n_classes for n rows
        labels = np.array([num for _ in range(n_row) for num in range(n_row)])
        labels = torch.LongTensor(labels).to(self.device)
        gen_imgs = self.generator(z, labels)
        save_image(gen_imgs.data, "output/gan/%d.png" % batches_done, nrow=n_row, normalize=True)
