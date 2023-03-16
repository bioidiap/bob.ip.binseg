import torch
import torchvision.transforms as T

from ..hrf.default_768 import dataset as _hrf

jitter = T.ColorJitter(saturation=0.3, brightness=0.3, contrast=0.3)
gray = T.Grayscale(num_output_channels=3)
train = []
for i in _hrf["train"]:
    j = i
    j[1] = i[1] + 0.01 * torch.randn_like(i[1])  # add gaussian noise
    j[1] = gray(j[1])
    # j[1] = jitter(j[1])
    train.append(j)
dataset = {
    "train": _hrf["train"],
    "test": _hrf["test"],
    "__train__": train,
    "__valid__": _hrf["train"],
}
dataset["__extra_valid__"] = [dataset["test"]]
