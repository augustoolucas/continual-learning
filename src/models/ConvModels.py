import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from pytorch_lightning.trainer import Trainer
from torch import nn
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
from cfg import cfg

import models.MLPModels as mlp
from models import base, utils


class cVAE(base.cVAE):

    def __init__(self, in_shape, latent_shape, n_classes):
        super().__init__(latent_shape)

        self.encoder = Encoder(in_shape, latent_shape)
        self.decoder = Decoder(in_shape, latent_shape, n_classes)


class cVAEGAN(cVAE):

    def __init__(self, in_shape, latent_shape, n_classes):
        super().__init__(in_shape=in_shape,
                         latent_shape=latent_shape,
                         n_classes=n_classes)

        self.discriminator = Discriminator(in_shape)

    def adversarial_loss(self, predictions, labels, reduction='mean'):
        return F.binary_cross_entropy_with_logits(predictions,
                                                  labels,
                                                  reduction=reduction)

    def configure_optimizers(self):
        vae_optimizer = torch.optim.Adam(list(self.encoder.parameters()) +
                                         list(self.decoder.parameters()),
                                         lr=float(cfg['lr_ae']))

        discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=float(cfg['lr_discriminator']))

        return vae_optimizer, discriminator_optimizer

    def vae_step(self, imgs, labels):
        one_hot_labels = one_hot(labels, num_classes=10)

        z, mu, var = self.encoder(imgs)
        recon_imgs = self.decoder(torch.cat([z, one_hot_labels], dim=1))

        real_labels = torch.ones(imgs.size(0), 1).type_as(imgs)

        disc_output = self.discriminator(recon_imgs)

        kl_loss = self.kl_loss(var, mu, reduction='none')
        recon_loss = self.pixelwise_loss(imgs, recon_imgs)
        disc_loss = self.adversarial_loss(disc_output, real_labels)

        z = torch.randn(size=z.shape).to(self.device)
        y = torch.randint(low=0, high=torch.max(labels), size=labels.shape)
        y = one_hot(y, num_classes=10).to(self.device)

        gen_imgs = self.decoder(torch.cat([z, y], dim=1))
        disc_output = self.discriminator(gen_imgs)

        disc_loss += self.adversarial_loss(disc_output, real_labels)

        cvae_loss = kl_loss + recon_loss + disc_loss

        return cvae_loss

    def discriminator_step(self, imgs, labels):
        one_hot_labels = one_hot(labels, num_classes=10)

        real_labels = torch.ones(imgs.size(0), 1).type_as(imgs)

        disc_output = self.discriminator(imgs)
        real_loss = self.adversarial_loss(disc_output, real_labels)

        fake_labels = torch.zeros(imgs.size(0), 1).type_as(imgs)

        # reconstruct images
        z, mu, var = self.encoder(imgs)
        recon_imgs = self.decoder(torch.cat([z, one_hot_labels], dim=1))

        disc_output = self.discriminator(recon_imgs)
        fake_loss = self.adversarial_loss(disc_output.detach(), fake_labels)

        # gen images
        z = torch.randn(size=z.shape).to(self.device)
        y = torch.randint(low=0, high=torch.max(labels), size=labels.shape)
        y = one_hot(y, num_classes=10).to(self.device)

        gen_imgs = self.decoder(torch.cat([z, y], dim=1))
        disc_output = self.discriminator(gen_imgs)

        fake_loss += self.adversarial_loss(disc_output, fake_labels)

        d_loss = real_loss + fake_loss

        return d_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, labels = batch

        if optimizer_idx == 0:
            return self.vae_step(imgs, labels)

        if optimizer_idx == 1:
            return self.discriminator_step(imgs, labels)


class Encoder(nn.Module):

    def __init__(self, in_shape, latent_shape):
        super().__init__()
        channels = in_shape[0] if in_shape[0] < in_shape[2] else in_shape[2]
        height = in_shape[0] if in_shape[0] > in_shape[2] else in_shape[1]

        assert channels in [1, 3]
        assert height in [28, 32]

        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True),
        )

        feat_map_dim = (128, 7, 7) if height == 28 else (128, 8, 8)
        self.mu = nn.Linear(np.prod(feat_map_dim), latent_shape)
        self.logvar = nn.Linear(np.prod(feat_map_dim), latent_shape)
        self.latent_shape = latent_shape

    def reparameterization(self, mu, logvar):
        std = torch.exp(logvar / 2)
        sampled_z = torch.normal(0, 1, size=(mu.size(0), self.latent_shape))
        sampled_z = sampled_z.to(mu.device)

        z = sampled_z * std + mu

        return z

    def forward(self, img):
        x = self.conv_block(img)
        x = x.view(x.shape[0], -1)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = self.reparameterization(mu, logvar)

        return z, mu, logvar


class Decoder(nn.Module):

    def __init__(self, out_shape, latent_shape, n_classes, use_label=True):
        super().__init__()

        channels = out_shape[0] if out_shape[0] < out_shape[2] else out_shape[2]
        height = out_shape[0] if out_shape[0] > out_shape[2] else out_shape[1]

        assert channels in [1, 3]
        assert height in [28, 32]

        self.feat_map_dim = (128, 7, 7) if height == 28 else (128, 8, 8)

        input_dim = latent_shape + n_classes if use_label else latent_shape

        self.linear_block = nn.Sequential(
            nn.Linear(input_dim, np.prod(self.feat_map_dim)),
            nn.BatchNorm1d(np.prod(self.feat_map_dim)),
            nn.ELU(inplace=True),
        )

        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, channels, kernel_size=2, stride=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.linear_block(z)
        x = self.conv_block(
            x.view(-1, 128, self.feat_map_dim[1], self.feat_map_dim[2]))

        return x


class Specific(nn.Module):

    def __init__(self, in_shape, specific_shape):
        super().__init__()

        channels = in_shape[0] if in_shape[0] < in_shape[2] else in_shape[2]
        height = in_shape[0] if in_shape[0] > in_shape[2] else in_shape[1]

        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, padding=1, stride=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
        )

        feat_map_dim = (64, 3, 3) if height == 28 else (64, 4, 4)

        self.linear_block = nn.Sequential(
            nn.Linear(np.prod(feat_map_dim), specific_shape),
            nn.ReLU(inplace=True))

    def forward(self, img):
        x = self.conv_block(img)
        x = x.view(x.shape[0], -1)
        x = self.linear_block(x)

        return x


class ClassifierSpecific(pl.LightningModule):

    def __init__(self, in_shape, latent_shape, specific_shape, n_classes, lr):
        super().__init__()

        self.lr = lr
        self.classifier = mlp.Classifier(latent_shape, specific_shape,
                                         n_classes)
        self.specific = Specific(in_shape, specific_shape)

    def forward(self, img):
        return self.encoder(img)

    def classification_loss(self, predictions, labels, reduction='mean'):
        return F.cross_entropy(predictions, labels, reduction=reduction)

    def configure_optimizers(self):
        return torch.optim.Adam(self.specific.parameters() +
                                self.classifier.parameters(),
                                lr=float(self.lr))

    def set_encoder(self, encoder):
        self.encoder = encoder

    def training_step(self, batch, batch_idx):
        imgs, labels = batch

        z, _, _ = self.encoder(imgs)
        specific_embedding = self.specific(imgs)
        classifier_output = self.classifier(specific_embedding, z.detach())
        classifier_loss = self.classification_loss(classifier_output, labels)

        return classifier_loss

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    def on_train_epoch_end(self):
        z = self.validation_z.to(self.device)
        y = torch.randint(low=0, high=9, size=(z.size(0), )).to(self.device)
        print(f'Gen Y: {y}')
        y = one_hot(y, num_classes=10)

        sample_imgs = self.decoder(torch.cat([z, y], dim=1))
        grid = torchvision.utils.make_grid(sample_imgs, nrow=3)
        torchvision.utils.save_image(grid, fp=f'img-{self.current_epoch}.jpg')


class Discriminator(nn.Module):

    def __init__(self, in_shape):
        super().__init__()

        channels = in_shape[0] if in_shape[0] < in_shape[2] else in_shape[2]
        height = in_shape[0] if in_shape[0] > in_shape[2] else in_shape[1]

        assert channels in [1, 3]
        assert height in [28, 32]

        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        feat_map_dim = (128, 20, 20) if height == 28 else (128, 8, 8)
        self.linear_block = nn.Sequential(nn.Linear(np.prod(feat_map_dim), 1))

    def forward(self, img):
        x = self.conv_block(img)
        x = x.view(x.shape[0], -1)
        x = self.linear_block(x)

        return x
