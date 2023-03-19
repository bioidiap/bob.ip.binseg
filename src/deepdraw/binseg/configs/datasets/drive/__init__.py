# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later
import torch
import torchvision.transforms as T


def _maker(protocol):
    from .....common.data.transforms import CenterCrop as ccrop
    from ....data.drive import dataset as raw
    from .. import make_dataset as mk

    return mk(raw.subsets(protocol), [ccrop((544, 544))])


def _maker_square(protocol, size):
    from .....common.data.transforms import Pad, Resize
    from ....data.drive import dataset as raw
    from .. import make_dataset as mk

    return mk(
        raw.subsets(protocol), [Pad((10, 1, 10, 0)), Resize((size, size))]
    )


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
