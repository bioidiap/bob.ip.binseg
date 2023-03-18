# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Semi-supervised pretrain dataset for the STARE dataset."""

import torch
import torchvision.transforms as T

from ..stare.ah_768 import dataset as _stare

jitter = T.ColorJitter(saturation=0.3)
gray = T.Grayscale(num_output_channels=3)

train = []
for i in _stare["train"]:
    j = i
    # j[1] = T.functional.adjust_sharpness(j[1],sharpness_factor=0.0) # sharpness
    j[1] = i[1] + 0.01 * torch.randn_like(i[1])  # add gaussian noise
    j[1] = gray(j[1])
    # j[1] = jitter(j[1])
    train.append(j)
dataset = {
    "train": train,
    "test": _stare["test"],
    "__train__": train,
    "__valid__": _stare["train"],
}
dataset["__extra_valid__"] = [dataset["test"]]
