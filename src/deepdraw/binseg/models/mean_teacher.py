import logging

import torch
import torch.nn
import torchvision.transforms as T
#choose the model to be used as the initialization network
from deepdraw.binseg.models import unet

# build Mean Teacher model
logger = logging.getLogger(__name__)
gray = T.Grayscale(num_output_channels=3)
jitter1 = T.ColorJitter(contrast=0.3)
jitter2 = T.ColorJitter(contrast=0.4)


def gauss1(x):
    return x + 0.01 * torch.randn_like(x)


def gauss2(x):
    return x + 0.02 * torch.randn_like(x)


def rotate1(x):
    return T.functional.rotate(x, angle=4.5)


def rotate2(x):
    return T.functional.rotate(x, angle=3)


def sharp1(x):
    return T.functional.adjust_sharpness(x, sharpness_factor=0.0)


def sharp2(x):
    return T.functional.adjust_sharpness(x, sharpness_factor=0.5)


class Mean_teacher(torch.nn.Module):
    def __init__(
        self,
        weight,
    ):
        super(Mean_teacher, self).__init__()
        self.T_model = unet.unet(pretrained_backbone=True, progress=True)
        self.S_model = unet.unet(pretrained_backbone=True, progress=True)

        self.weight = weight
        if weight is None:
            logger.info(f"Model is not pretrained")
        else:
            #initialize the model with pretrained weights
            logger.info(f"Loading pretrained model from {weight}")
            input_model = unet.unet()
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
        x_t = gray(gauss1((x)))
        x_s = gray(gauss2((x)))
        y_s = self.S_model(x_s)
        y_t = self.T_model(x_t)
        return y_s, y_t


def mean_teacher(weight):
    model = Mean_teacher(weight)
    model.name = "mean_teacher"
    return model
