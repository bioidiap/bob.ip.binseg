#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from collections import OrderedDict

import torch
import torch.nn
from torchvision.models.mobilenet import InvertedResidual

from .backbones.mobilenetv2 import mobilenet_v2_for_segmentation


class DecoderBlock(torch.nn.Module):
    """
    Decoder block: upsample and concatenate with features maps from the encoder part
    """

    def __init__(
        self, up_in_c, x_in_c, upsamplemode="bilinear", expand_ratio=0.15
    ):
        super().__init__()
        self.upsample = torch.nn.Upsample(
            scale_factor=2, mode=upsamplemode, align_corners=False
        )  # H, W -> 2H, 2W
        self.ir1 = InvertedResidual(
            up_in_c + x_in_c,
            (x_in_c + up_in_c) // 2,
            stride=1,
            expand_ratio=expand_ratio,
        )

    def forward(self, up_in, x_in):
        up_out = self.upsample(up_in)
        cat_x = torch.cat([up_out, x_in], dim=1)
        x = self.ir1(cat_x)
        return x


class LastDecoderBlock(torch.nn.Module):
    def __init__(self, x_in_c, upsamplemode="bilinear", expand_ratio=0.15):
        super().__init__()
        self.upsample = torch.nn.Upsample(
            scale_factor=2, mode=upsamplemode, align_corners=False
        )  # H, W -> 2H, 2W
        self.ir1 = InvertedResidual(
            x_in_c, 1, stride=1, expand_ratio=expand_ratio
        )

    def forward(self, up_in, x_in):
        up_out = self.upsample(up_in)
        cat_x = torch.cat([up_out, x_in], dim=1)
        x = self.ir1(cat_x)
        return x


class M2UNet(torch.nn.Module):
    """
    M2U-Net head module

    Parameters
    ----------
    in_channels_list : list
        number of channels for each feature map that is returned from backbone
    """

    def __init__(
        self, in_channels_list=None, upsamplemode="bilinear", expand_ratio=0.15
    ):
        super(M2UNet, self).__init__()

        # Decoder
        self.decode4 = DecoderBlock(96, 32, upsamplemode, expand_ratio)
        self.decode3 = DecoderBlock(64, 24, upsamplemode, expand_ratio)
        self.decode2 = DecoderBlock(44, 16, upsamplemode, expand_ratio)
        self.decode1 = LastDecoderBlock(33, upsamplemode, expand_ratio)

        # initilaize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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
        tensor : :py:class:`torch.Tensor`
        """
        decode4 = self.decode4(x[5], x[4])  # 96, 32
        decode3 = self.decode3(decode4, x[3])  # 64, 24
        decode2 = self.decode2(decode3, x[2])  # 44, 16
        decode1 = self.decode1(decode2, x[1])  # 30, 3

        return decode1


def m2unet(pretrained_backbone=True, progress=True):
    """Builds M2U-Net for segmentation by adding backbone and head together


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
        Network model for M2U-Net (segmentation)

    """

    backbone = mobilenet_v2_for_segmentation(
        pretrained=pretrained_backbone,
        progress=progress,
        return_features=[1, 3, 6, 13],
    )
    head = M2UNet(in_channels_list=[16, 24, 32, 96])

    order = [("backbone", backbone), ("head", head)]
    if pretrained_backbone:
        from .normalizer import TorchVisionNormalizer
        order = [("normalizer", TorchVisionNormalizer())] + order

    model = torch.nn.Sequential(OrderedDict(order))
    model.name = "m2unet"
    return model
