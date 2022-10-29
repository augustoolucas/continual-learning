import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from cfg import cfg
from torch import nn
from torch.nn.functional import one_hot

from models import utils


class cVAE(pl.LightningModule):

    def __init__(self, latent_shape):
        super().__init__()

        self.encoder: nn.Module
        self.decoder: nn.Module
        self.validation_z = torch.randn(9, latent_shape)

    def forward(self, img):
        return self.encoder(img)

    def pixelwise_loss(self, img1, img2, reduction="mean"):
        return F.mse_loss(img1, img2, reduction=reduction)

    def kl_loss(self, var, mu, reduction="mean"):
        return utils.kl_loss(var, mu, reduction)

    def configure_optimizers(self):
        vae_optimizer = torch.optim.Adam(list(self.encoder.parameters()) +
                                         list(self.decoder.parameters()),
                                         lr=float(cfg["lr_ae"]))

        return vae_optimizer

    def training_step(self, batch, batch_idx):
        imgs, labels = batch

        one_hot_labels = one_hot(labels, num_classes=10)

        z, mu, var = self.encoder(imgs)
        recon_imgs = self.decoder(torch.cat([z, one_hot_labels], dim=1))

        kl_loss = self.kl_loss(var, mu, reduction="none")
        recon_loss = self.pixelwise_loss(imgs, recon_imgs)

        cvae_loss = kl_loss + recon_loss

        return cvae_loss

    def on_train_epoch_end(self):
        z = self.validation_z.to(self.device)
        y = torch.randint(low=0, high=9, size=(z.size(0), )).to(self.device)
        print(f"Gen Y: {y}")
        y = one_hot(y, num_classes=10)

        sample_imgs = self.decoder(torch.cat([z, y], dim=1))
        grid = torchvision.utils.make_grid(sample_imgs, nrow=3)
        torchvision.utils.save_image(grid, fp=f"img-{self.current_epoch}.jpg")
