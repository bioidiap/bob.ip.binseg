#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import OrderedDict

import torch
import torch.nn

from .backbones.vgg import vgg16_for_segmentation

from .make_layers import UpsampleCropBlock
from .driu import ConcatFuseBlock


class DRIUOD(torch.nn.Module):
    """
    DRIU for optic disc segmentation head module

    Parameters
    ----------
    in_channels_list : list
        number of channels for each feature map that is returned from backbone
    """

    def __init__(self, in_channels_list=None):
        super(DRIUOD, self).__init__()
        in_upsample2, in_upsample_4, in_upsample_8, in_upsample_16 = (
            in_channels_list
        )

        self.upsample2 = UpsampleCropBlock(in_upsample2, 16, 4, 2, 0)
        # Upsample layers
        self.upsample4 = UpsampleCropBlock(in_upsample_4, 16, 8, 4, 0)
        self.upsample8 = UpsampleCropBlock(in_upsample_8, 16, 16, 8, 0)
        self.upsample16 = UpsampleCropBlock(in_upsample_16, 16, 32, 16, 0)

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
        upsample2 = self.upsample2(x[1], hw)  # side-multi2-up
        upsample4 = self.upsample4(x[2], hw)  # side-multi3-up
        upsample8 = self.upsample8(x[3], hw)  # side-multi4-up
        upsample16 = self.upsample16(x[4], hw)  # side-multi5-up
        out = self.concatfuse(upsample2, upsample4, upsample8, upsample16)
        return out


def driu_od(pretrained_backbone=True, progress=True):
    """Builds DRIU for Optical Disc by adding backbone and head together

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
        Network model for DRIU (optic disc segmentation)

    """

    backbone = vgg16_for_segmentation(
        pretrained=pretrained_backbone,
        progress=progress,
        return_features=[8, 14, 22, 29],
    )
    head = DRIUOD([128, 256, 512, 512])

    order = [("backbone", backbone), ("head", head)]
    if pretrained_backbone:
        from .normalizer import TorchVisionNormalizer

        order = [("normalizer", TorchVisionNormalizer())] + order

    model = torch.nn.Sequential(OrderedDict(order))
    model.name = "driu-od"
    return model
