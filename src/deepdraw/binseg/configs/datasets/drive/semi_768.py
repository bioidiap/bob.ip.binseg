# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Semi-supervised pretrain dataset for the DRIVE dataset."""
import torch
import torchvision.transforms as T

from ..drive.default_768 import dataset as _drive

jitter = T.ColorJitter(hue=0.05)


def rotate(x):
    return T.functional.rotate(x, angle=4.5)


gray = T.Grayscale(num_output_channels=3)


def _transform(dataset):
    train = []
    for i in dataset:
        j = i
        j[1] = i[1] + 0.01 * torch.randn_like(i[1])  # add gaussian noise
        j[1] = gray(j[1])
        # j[1] = jitter(j[1])
        # j[1] = rotate(j[1])
        train.append(j)
    return train


dataset = {
    "train": _drive["train"],
    "test": _drive["test"],
    "__train__": _transform(_drive["train"]),
    "__valid__": _drive["train"],
}
dataset["__extra_valid__"] = [dataset["test"]]
