import torch
import torchvision.transforms as T

from ..iostar.vessel_768 import dataset as _iostar

jitter = T.ColorJitter(saturation=0.3)
gray = T.Grayscale(num_output_channels=3)
train = []
for i in _iostar["train"]:
    j = i
    j[1] = i[1] + 0.01 * torch.randn_like(i[1])  # add gaussian noise
    j[1] = gray(j[1])
    # j[1] = jitter(j[1])
    train.append(j)
dataset = {
    "train": _iostar["train"],
    "test": _iostar["test"],
    "__train__": train,
    "__valid__": _iostar["train"],
}
dataset["__extra_valid__"] = [dataset["test"]]