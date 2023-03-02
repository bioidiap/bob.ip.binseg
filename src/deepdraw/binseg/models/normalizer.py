# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A network model that prefixes a z-normalization step to any other module."""


import torch
import torch.nn


class TorchVisionNormalizer(torch.nn.Module):
    """A simple normalizer that applies the standard torchvision normalization.

    This module does not learn.

    The values applied in this "prefix" operator are defined at
    https://pytorch.org/docs/stable/torchvision/models.html, and are as
    follows:

    * ``mean``: ``[0.485, 0.456, 0.406]``,
    * ``std``: ``[0.229, 0.224, 0.225]``
    """

    def __init__(self):
        super().__init__()
        mean = torch.as_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = torch.as_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.name = "torchvision-normalizer"

    def forward(self, inputs):
        return inputs.sub(self.mean).div(self.std)
