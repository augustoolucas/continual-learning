import pytorch_lightning as pl

import models
from cfg import test_cfg, train_cfg
from data import utils

benchmark = utils.load_data("MNIST")

encoder = models.MLPModels.Encoder(img_shape=(1, 28, 28), latent_dim=32)
decoder = models.MLPModels.Decoder(img_shape=(1, 28, 28),
                                   latent_dim=32,
                                   n_classes=10)

specific = models.MLPModels.Specific(img_shape=(1, 28, 28), specific_size=20)
classifier = models.MLPModels.Classifier(latent_size=32,
                                         specific_size=20,
                                         n_classes=10)

ircl = models.IRCL.IRCL(specific, classifier, encoder, decoder)

train_stream = benchmark.train_stream
test_stream = benchmark.test_stream

# Continual Learning

for train_exp, test_exp in zip(train_stream, test_stream):
    train_loader = utils.get_dataloader(train_exp.dataset,
                                        train_cfg["batch_size"])
    test_loader = utils.get_dataloader(test_exp.dataset,
                                       test_cfg["batch_size"],
                                       shuffle=False)
    trainer = pl.Trainer(max_epochs=train_cfg["epochs"], accelerator="gpu")
    trainer.fit(ircl, train_loader)
    trainer.test(dataloaders=test_loader, ckpt_path="best")

test_loader = utils.get_dataloader(benchmark.original_test_dataset,
                                   test_cfg["batch_size"],
                                   shuffle=False)

trainer.test(dataloaders=test_loader, ckpt_path="best")

# End to end
train_loader = utils.get_dataloader(benchmark.original_train_dataset,
                                    train_cfg["batch_size"])

trainer = pl.Trainer(max_epochs=train_cfg["epochs"], accelerator="gpu")
trainer.fit(ircl, train_loader)
trainer.test(dataloaders=test_loader, ckpt_path="best")
