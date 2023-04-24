# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import logging

import torch
import torch.nn
import torchvision.transforms as T

# choose the model to be used as the initialization network
from deepdraw.binseg.models import unet

# build Mean Teacher model
logger = logging.getLogger(__name__)
gray = T.Grayscale(num_output_channels=3)
jitter1 = T.ColorJitter(contrast=0.3)
jitter2 = T.ColorJitter(contrast=0.4)


def gauss(x, std):
    return x + std * torch.randn_like(x)


def rotate(x, ang):
    return T.functional.rotate(x, angle=ang)


class Mean_teacher(torch.nn.Module):
    def __init__(
        self,
        weight,
    ):
        super().__init__()
        self.T_model = unet.unet(pretrained_backbone=True, progress=True)
        self.S_model = unet.unet(pretrained_backbone=True, progress=True)

        self.weight = weight
        input_model = unet.unet()
        if weight is None:
            logger.info("Model is not pretrained")
        else:
            # initialize the model with pretrained weights
            logger.info(f"Loading pretrained model from {weight}")
            input_model.load_state_dict(torch.load(weight), strict=False)
            for T_param, input_param in zip(
                self.T_model.parameters(), input_model.parameters()
            ):
                T_param.data.copy_(input_param.data)
            for S_param, input_param in zip(
                self.S_model.parameters(), input_model.parameters()
            ):
                S_param.data.copy_(input_param.data)

    # forward should return the output of both models
    def forward(self, x):
        self.x = x
        # data augmentation on input image
        x_t = gray(gauss(x, 0.01))
        x_s = gray(gauss(x, 0.02))
        y_s = self.S_model(x_s)
        y_t = self.T_model(x_t)
        return y_s, y_t


def mean_teacher(weight):
    model = Mean_teacher(weight)
    model.name = "mean_teacher"
    return model
