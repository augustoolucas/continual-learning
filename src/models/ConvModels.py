import numpy as np
import torch
from torch import nn

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
        sampled_z = torch.Tensor(
            np.random.normal(0, 1, (mu.size(0), latent_dim)))
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
