import torch
import torchvision.transforms as T

from ..drive.default_768 import dataset as _drive

from .. import make_dataset as mk

jitter = T.ColorJitter(hue=0.05)


def rotate(x):
    return T.functional.rotate(x, angle=4.5)


gray = T.Grayscale(num_output_channels=3)
train = []
for i in _drive["train"]:
    j = i
    j[1] = i[1] + 0.01 * torch.randn_like(i[1])  # add gaussian noise
    j[1] = gray(j[1])
    # j[1] = jitter(j[1])
    # j[1] = rotate(j[1])
    train.append(j)
dataset = {
    "train": _drive["train"],
    "test": _drive["test"],
    "__train__": train,
    "__valid__": _drive["train"],
}
dataset["__extra_valid__"] = [dataset["test"]]
