import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from models import base


class cVAE(base.cVAE):

    def __init__(self, in_shape, latent_shape, n_classes):
        super().__init__(latent_shape)

        self.encoder = Encoder(in_shape, latent_shape)
        self.decoder = Decoder(in_shape, latent_shape, n_classes)


class Encoder(nn.Module):

    def __init__(self, img_shape, latent_dim):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 40),
            nn.BatchNorm1d(40),
            nn.ReLU(inplace=True),
        )
        self.mu = nn.Linear(40, latent_dim)
        self.logvar = nn.Linear(40, latent_dim)
        self.latent_dim = latent_dim

    def reparameterization(self, mu, logvar):
        std = torch.exp(logvar / 2)
        sampled_z = torch.randn(size=(mu.size(0), self.latent_dim))
        sampled_z = sampled_z.to(mu.device)
        z = sampled_z * std + mu

        return z

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        x = self.model(img_flat)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = self.reparameterization(mu, logvar)

        return z, mu, logvar


class Decoder(nn.Module):

    def __init__(self,
                 img_shape,
                 latent_dim,
                 n_classes,
                 use_label=True):
        super(Decoder, self).__init__()
        # conditional generation

        if use_label:
            input_dim = latent_dim + n_classes
        else:
            input_dim = latent_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, 40),
            nn.BatchNorm1d(40),
            nn.ReLU(inplace=True),
            nn.Linear(40, int(np.prod(img_shape))),
            nn.Sigmoid(),
        )
        self.img_shape = img_shape

    def forward(self, z):
        img_flat = self.model(z)
        img = img_flat.view(img_flat.shape[0], *self.img_shape)

        return img


class Specific(nn.Module):

    def __init__(self, img_shape, specific_size):
        super(Specific, self).__init__()

        # specific module
        self.specific = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), specific_size),
            nn.ReLU(inplace=True),
        )

    def forward(self, imgs):
        x = self.specific(imgs.view(imgs.shape[0], -1))

        return x


class Classifier(nn.Module):

    def __init__(self,
                 latent_size,
                 specific_size,
                 n_classes,
                 classification_n_hidden=40,
                 softmax=False):
        super(Classifier, self).__init__()

        self.classifier_layer = nn.Sequential(
            nn.Linear(specific_size + latent_size, classification_n_hidden),
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
