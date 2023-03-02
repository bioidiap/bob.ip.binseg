# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from collections import OrderedDict

import torch.nn

from .backbones.vgg import vgg16_for_segmentation
from .make_layers import UnetBlock, conv_with_kaiming_uniform


class UNet(torch.nn.Module):
    """UNet head module.

    Parameters
    ----------
    in_channels_list : list
                        number of channels for each feature map that is returned from backbone
    """

    def __init__(self, in_channels_list=None, pixel_shuffle=False):
        super().__init__()
        # number of channels
        c_decode1, c_decode2, c_decode3, c_decode4, c_decode5 = in_channels_list

        # build layers
        self.decode4 = UnetBlock(
            c_decode5, c_decode4, pixel_shuffle, middle_block=True
        )
        self.decode3 = UnetBlock(c_decode4, c_decode3, pixel_shuffle)
        self.decode2 = UnetBlock(c_decode3, c_decode2, pixel_shuffle)
        self.decode1 = UnetBlock(c_decode2, c_decode1, pixel_shuffle)
        self.final = conv_with_kaiming_uniform(c_decode1, 1, 1)

    def forward(self, x):
        """
        Parameters
        ----------
        x : list
            list of tensors as returned from the backbone network.
            First element: height and width of input image.
            Remaining elements: feature maps for each feature level.
        """
        # NOTE: x[0]: height and width of input image not needed in U-Net architecture
        decode4 = self.decode4(x[5], x[4])
        decode3 = self.decode3(decode4, x[3])
        decode2 = self.decode2(decode3, x[2])
        decode1 = self.decode1(decode2, x[1])
        out = self.final(decode1)
        return out


def unet(pretrained_backbone=True, progress=True):
    """Builds U-Net segmentation network by adding backbone and head together.

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
        Network model for U-Net
    """

    backbone = vgg16_for_segmentation(
        pretrained=pretrained_backbone,
        progress=progress,
        return_features=[3, 8, 14, 22, 29],
    )
    head = UNet([64, 128, 256, 512, 512], pixel_shuffle=False)

    order = [("backbone", backbone), ("head", head)]
    if pretrained_backbone:
        from .normalizer import TorchVisionNormalizer

        order = [("normalizer", TorchVisionNormalizer())] + order

    model = torch.nn.Sequential(OrderedDict(order))
    model.name = "unet"
    return model
