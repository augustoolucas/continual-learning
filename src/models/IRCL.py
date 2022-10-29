import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from cfg import train_cfg
from torchmetrics.functional import accuracy

from models import utils


class IRCL(pl.LightningModule):

    def __init__(self, specific, classifier, encoder, decoder):
        super().__init__()
        self.specific = specific
        self.classifier = classifier
        self.encoder = encoder
        self.decoder = decoder

    def classification_loss(self, input, target, reduction='mean'):
        return F.cross_entropy(input, target, reduction=reduction)

    def pixelwise_loss(self, img1, img2, reduction='mean'):
        return F.mse_loss(img1, img2, reduction=reduction)

    def kl_loss(self, var, mu, reduction='mean'):
        return utils.kl_loss(var, mu, reduction)

    def configure_optimizers(self):
        cvae_optimizer = torch.optim.Adam(list(self.encoder.parameters()) +
                                          list(self.decoder.parameters()),
                                          lr=float(train_cfg['lr_cvae']))

        cls_optimizer = torch.optim.Adam(list(self.specific.parameters()) +
                                         list(self.classifier.parameters()),
                                         lr=float(train_cfg['lr_cls']))

        return cvae_optimizer, cls_optimizer

    def cvae_step(self, imgs, labels):
        one_hot_labels = F.one_hot(labels, num_classes=10)

        z, mu, var = self.encoder(imgs)
        recon_imgs = self.decoder(torch.cat([z, one_hot_labels], dim=1))

        kl_loss = self.kl_loss(var, mu, reduction='none')
        recon_loss = self.pixelwise_loss(imgs, recon_imgs)

        cvae_loss = kl_loss + recon_loss

        return cvae_loss

    def classifier_step(self, imgs, labels):
        z, _, _ = self.encoder(imgs)
        specific_embedding = self.specific(imgs)
        outputs = self.classifier(specific_embedding, z.detach())
        c_loss = self.classification_loss(outputs, labels)

        return c_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, labels, _ = batch

        if optimizer_idx == 0:
            return self.cvae_step(imgs, labels)

        if optimizer_idx == 1:
            return self.classifier_step(imgs, labels)

    def test_step(self, batch, batch_idx):
        imgs, labels, _ = batch

        z, _, _ = self.encoder(imgs)
        specific_embedding = self.specific(imgs)
        outputs = self.classifier(specific_embedding, z.detach())
        classification_loss = self.classification_loss(outputs, labels)
        acc = accuracy(outputs, labels)
        metrics = {"Test Loss": classification_loss.item(),
                   "Test Accuracy": acc.item()}
        self.log_dict(metrics)

        return metrics
