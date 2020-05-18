#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import OrderedDict

import torch.nn

from .make_layers import (
    conv_with_kaiming_uniform,
    convtrans_with_kaiming_uniform,
    PixelShuffle_ICNR,
    UnetBlock,
)

from .backbones.resnet import resnet50_for_segmentation


class ResUNet(torch.nn.Module):
    """UNet head module for ResNet backbones

    Parameters
    ----------

    in_channels_list : :py:class:`list`, Optional
        number of channels for each feature map that is returned from backbone

    pixel_shuffle : :py:class:`bool`, Optional
        if should use pixel shuffling instead of pooling

    """

    def __init__(self, in_channels_list=None, pixel_shuffle=False):
        super(ResUNet, self).__init__()
        # number of channels
        c_decode1, c_decode2, c_decode3, c_decode4, c_decode5 = in_channels_list
        # number of channels for last upsampling operation
        c_decode0 = (c_decode1 + c_decode2 // 2) // 2

        # build layers
        self.decode4 = UnetBlock(c_decode5, c_decode4, pixel_shuffle)
        self.decode3 = UnetBlock(c_decode4, c_decode3, pixel_shuffle)
        self.decode2 = UnetBlock(c_decode3, c_decode2, pixel_shuffle)
        self.decode1 = UnetBlock(c_decode2, c_decode1, pixel_shuffle)
        if pixel_shuffle:
            self.decode0 = PixelShuffle_ICNR(c_decode0, c_decode0)
        else:
            self.decode0 = convtrans_with_kaiming_uniform(
                c_decode0, c_decode0, 2, 2
            )
        self.final = conv_with_kaiming_uniform(c_decode0, 1, 1)

    def forward(self, x):
        """
        Parameters
        ----------
        x : list
                list of tensors as returned from the backbone network.
                First element: height and width of input image.
                Remaining elements: feature maps for each feature level.
        """
        # NOTE: x[0]: height and width of input image not needed in U-Net
        # architecture
        decode4 = self.decode4(x[5], x[4])
        decode3 = self.decode3(decode4, x[3])
        decode2 = self.decode2(decode3, x[2])
        decode1 = self.decode1(decode2, x[1])
        decode0 = self.decode0(decode1)
        out = self.final(decode0)
        return out


def resunet50(pretrained_backbone=True, progress=True):
    """Builds Residual-U-Net-50 by adding backbone and head together

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
        Network model for Residual U-Net 50

    """

    backbone = resnet50_for_segmentation(
        pretrained=pretrained_backbone,
        progress=progress,
        return_features=[2, 4, 5, 6, 7],
    )
    head = ResUNet([64, 256, 512, 1024, 2048], pixel_shuffle=False)

    order = [("backbone", backbone), ("head", head)]
    if pretrained_backbone:
        from .normalizer import TorchVisionNormalizer

        order = [("normalizer", TorchVisionNormalizer())] + order

    model = torch.nn.Sequential(OrderedDict(order))
    model.name = "resunet50"
    return model
