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

from models import utils


class VAEGAN(pl.LightningModule):

    def __init__(self, in_shape, latent_shape, n_classes, cfg):
        super().__init__()

        self.encoder = Encoder(in_shape, latent_shape)
        self.decoder = Decoder(in_shape, latent_shape, n_classes)
        self.discriminator = Discriminator(in_shape)
        self.cfg = cfg
        self.validation_z = torch.randn(8, self.encoder.latent_dim)

    def forward(self, img):
        return self.encoder(img)

    def pixelwise_loss(self, img1, img2, reduction='mean'):
        return F.mse_loss(img1, img2, reduction=reduction)

    def adversarial_loss(self, predictions, labels, reduction='mean'):
        return F.binary_cross_entropy_with_logits(predictions,
                                                  labels,
                                                  reduction=reduction)

    def kl_loss(self, var, mu, reduction='mean'):
        return utils.kl_loss(var, mu, reduction)

    def configure_optimizers(self):
        vae_optimizer = torch.optim.Adam(list(self.encoder.parameters()) +
                                         list(self.decoder.parameters()),
                                         lr=float(self.cfg['lr_ae']))

        discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=float(self.cfg['lr_discriminator']))

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

    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        dataset = MNIST('./datasets',
                        train=True,
                        download=True,
                        transform=transform)

        return DataLoader(dataset,
                          batch_size=256,
                          num_workers=8,
                          pin_memory=True,
                          shuffle=True)

    def on_train_epoch_end(self):
        z = self.validation_z.to(self.device)
        y = torch.randint(low=0, high=9, size=(z.size(0), )).to(self.device)
        print(f'Gen Y: {y}')
        y = one_hot(y, num_classes=10)

        sample_imgs = self.decoder(torch.cat([z, y], dim=1))
        grid = torchvision.utils.make_grid(sample_imgs)
        torchvision.utils.save_image(grid, fp=f'img-{self.current_epoch}.jpg')


class Encoder(nn.Module):

    def __init__(self, in_shape, out_shape):
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
        self.mu = nn.Linear(np.prod(feat_map_dim), out_shape)
        self.logvar = nn.Linear(np.prod(feat_map_dim), out_shape)
        self.latent_dim = out_shape

    def reparameterization(self, mu, logvar, latent_dim):
        std = torch.exp(logvar / 2)
        sampled_z = torch.normal(0, 1, size=(mu.size(0), latent_dim))
        sampled_z = sampled_z.to(mu.device)

        z = sampled_z * std + mu

        return z

    def forward(self, img):
        x = self.conv_block(img)
        x = x.view(x.shape[0], -1)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = self.reparameterization(mu, logvar, self.latent_dim)

        return z, mu, logvar


class Decoder(nn.Module):

    def __init__(self, in_shape, latent_dim, n_classes, use_label=True):
        super().__init__()

        channels = in_shape[0] if in_shape[0] < in_shape[2] else in_shape[2]
        height = in_shape[0] if in_shape[0] > in_shape[2] else in_shape[1]

        assert channels in [1, 3]
        assert height in [28, 32]

        self.feat_map_dim = (128, 7, 7) if height == 28 else (128, 8, 8)

        input_dim = latent_dim + n_classes if use_label else latent_dim

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

        self.in_shape = in_shape

    def forward(self, z):
        x = self.linear_block(z)
        x = self.conv_block(
            x.view(-1, 128, self.feat_map_dim[1], self.feat_map_dim[2]))

        return x


class Specific(nn.Module):

    def __init__(self, in_shape, specific_size):
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
            nn.Linear(np.prod(feat_map_dim), specific_size),
            nn.ReLU(inplace=True))

    def forward(self, img):
        x = self.conv_block(img)
        x = x.view(x.shape[0], -1)
        x = self.linear_block(x)

        return x


class Classifier(nn.Module):

    def __init__(self,
                 invariant_size,
                 specific_size,
                 classification_n_hidden,
                 n_classes,
                 softmax=False):
        super(Classifier, self).__init__()

        # classification module
        self.classifier_layer = nn.Sequential(
            nn.Linear(specific_size + invariant_size, classification_n_hidden),
            nn.ReLU(inplace=True),
        )

        modules = [nn.Linear(classification_n_hidden, n_classes)]

        if softmax:
            modules.append(nn.Softmax(dim=1))

        self.output = nn.Sequential(*modules)

    def forward(self, discriminative, invariant):
        x = self.classifier_layer(torch.cat([discriminative, invariant],
                                            dim=1))
        logits = self.output(x)

        return logits


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
