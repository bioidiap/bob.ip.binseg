# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Semi-supervised pretrain dataset for the HRF dataset."""

import torch
import torchvision.transforms as T

from ..hrf.default_768 import dataset as _hrf

jitter = T.ColorJitter(saturation=0.3, brightness=0.3, contrast=0.3)
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
    "train": _hrf["train"],
    "test": _hrf["test"],
    "__train__": _transform(_hrf["train"]),
    "__valid__": _hrf["train"],
}
dataset["__extra_valid__"] = [dataset["test"]]
