#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import OrderedDict

import torch
import torch.nn
from .backbones.vgg import vgg16_bn_for_segmentation

from .make_layers import conv_with_kaiming_uniform, UpsampleCropBlock


class ConcatFuseBlock(torch.nn.Module):
    """
    Takes in four feature maps with 16 channels each, concatenates them
    and applies a 1x1 convolution with 1 output channel.
    """

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Sequential(
            conv_with_kaiming_uniform(4 * 16, 1, 1, 1, 0),
            torch.nn.BatchNorm2d(1),
        )

    def forward(self, x1, x2, x3, x4):
        x_cat = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.conv(x_cat)
        return x


class DRIUBN(torch.nn.Module):
    """
    DRIU with Batch-Normalization head module

    Based on paper by [MANINIS-2016]_.

    Parameters
    ----------
    in_channels_list : list
        number of channels for each feature map that is returned from backbone
    """

    def __init__(self, in_channels_list=None):
        super(DRIUBN, self).__init__()
        (
            in_conv_1_2_16,
            in_upsample2,
            in_upsample_4,
            in_upsample_8,
        ) = in_channels_list

        self.conv1_2_16 = torch.nn.Conv2d(in_conv_1_2_16, 16, 3, 1, 1)
        # Upsample layers
        self.upsample2 = UpsampleCropBlock(in_upsample2, 16, 4, 2, 0)
        self.upsample4 = UpsampleCropBlock(in_upsample_4, 16, 8, 4, 0)
        self.upsample8 = UpsampleCropBlock(in_upsample_8, 16, 16, 8, 0)

        # Concat and Fuse
        self.concatfuse = ConcatFuseBlock()

    def forward(self, x):
        """
        Parameters
        ----------
        x : list
            list of tensors as returned from the backbone network.
            First element: height and width of input image.
            Remaining elements: feature maps for each feature level.

        Returns
        -------
        :py:class:`torch.Tensor`
        """
        hw = x[0]
        conv1_2_16 = self.conv1_2_16(x[1])  # conv1_2_16
        upsample2 = self.upsample2(x[2], hw)  # side-multi2-up
        upsample4 = self.upsample4(x[3], hw)  # side-multi3-up
        upsample8 = self.upsample8(x[4], hw)  # side-multi4-up
        out = self.concatfuse(conv1_2_16, upsample2, upsample4, upsample8)
        return out


def driu_bn(pretrained_backbone=True, progress=True):
    """Builds DRIU with batch-normalization by adding backbone and head together

    Parameters
    ----------

    pretrained_backbone : :py:class:`bool`, Optional
        If set to ``True``, then loads a pre-trained version of the backbone
        (not the head) for the DRIU network using VGG-16 trained for ImageNet
        classification.

    progress : :py:class:`bool`, Optional
        If set to ``True``, and you decided to use a ``pretrained_backbone``,
        then, shows a progress bar of the backbone model downloading if
        download is necesssary.


    Returns
    -------

    module : :py:class:`torch.nn.Module`
        Network model for DRIU (vessel segmentation) using batch normalization

    """

    backbone = vgg16_bn_for_segmentation(
        pretrained=False, return_features=[5, 12, 19, 29]
    )
    head = DRIUBN([64, 128, 256, 512])

    order = [("backbone", backbone), ("head", head)]
    if pretrained_backbone:
        from .normalizer import TorchVisionNormalizer
        order = [("normalizer", TorchVisionNormalizer())] + order

    model = torch.nn.Sequential(OrderedDict(order))
    model.name = "driu-bn"
    return model
