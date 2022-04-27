import models
from data import utils

benchmark = utils.load_data('SplitMNIST')

train_stream = benchmark.train_stream

model = models.Encoder((1, 28, 28), 32)

for experience in train_stream:
    for img, label, task_id in experience.dataset:
        model(img.unsqueeze(0))
        breakpoint()
