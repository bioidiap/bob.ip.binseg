# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Little W-Net.

Code was originally developed by Adrian Galdran
(https://github.com/agaldran/lwnet), loosely inspired on
https://github.com/jvanvugt/pytorch-unet

It is based on two simple U-Nets with 3 layers concatenated to each other.  The
first U-Net produces a segmentation map that is used by the second to better
guide segmentation.

Reference: [GALDRAN-2020]_
"""


import torch
import torch.nn


def _conv1x1(in_planes, out_planes, stride=1):
    return torch.nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False
    )


class ConvBlock(torch.nn.Module):
    def __init__(self, in_c, out_c, k_sz=3, shortcut=False, pool=True):
        """pool_mode can be False (no pooling) or True ('maxpool')"""

        super().__init__()
        if shortcut is True:
            self.shortcut = torch.nn.Sequential(
                _conv1x1(in_c, out_c), torch.nn.BatchNorm2d(out_c)
            )
        else:
            self.shortcut = False
        pad = (k_sz - 1) // 2

        block = []
        if pool:
            self.pool = torch.nn.MaxPool2d(kernel_size=2)
        else:
            self.pool = False

        block.append(
            torch.nn.Conv2d(in_c, out_c, kernel_size=k_sz, padding=pad)
        )
        block.append(torch.nn.ReLU())
        block.append(torch.nn.BatchNorm2d(out_c))

        block.append(
            torch.nn.Conv2d(out_c, out_c, kernel_size=k_sz, padding=pad)
        )
        block.append(torch.nn.ReLU())
        block.append(torch.nn.BatchNorm2d(out_c))

        self.block = torch.nn.Sequential(*block)

    def forward(self, x):
        if self.pool:
            x = self.pool(x)
        out = self.block(x)
        if self.shortcut:
            return out + self.shortcut(x)
        else:
            return out


class UpsampleBlock(torch.nn.Module):
    def __init__(self, in_c, out_c, up_mode="transp_conv"):
        super().__init__()
        block = []
        if up_mode == "transp_conv":
            block.append(
                torch.nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
            )
        elif up_mode == "up_conv":
            block.append(
                torch.nn.Upsample(
                    mode="bilinear", scale_factor=2, align_corners=False
                )
            )
            block.append(torch.nn.Conv2d(in_c, out_c, kernel_size=1))
        else:
            raise Exception("Upsampling mode not supported")

        self.block = torch.nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class ConvBridgeBlock(torch.nn.Module):
    def __init__(self, channels, k_sz=3):
        super().__init__()
        pad = (k_sz - 1) // 2
        block = []

        block.append(
            torch.nn.Conv2d(channels, channels, kernel_size=k_sz, padding=pad)
        )
        block.append(torch.nn.ReLU())
        block.append(torch.nn.BatchNorm2d(channels))

        self.block = torch.nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UpConvBlock(torch.nn.Module):
    def __init__(
        self,
        in_c,
        out_c,
        k_sz=3,
        up_mode="up_conv",
        conv_bridge=False,
        shortcut=False,
    ):
        super().__init__()
        self.conv_bridge = conv_bridge

        self.up_layer = UpsampleBlock(in_c, out_c, up_mode=up_mode)
        self.conv_layer = ConvBlock(
            2 * out_c, out_c, k_sz=k_sz, shortcut=shortcut, pool=False
        )
        if self.conv_bridge:
            self.conv_bridge_layer = ConvBridgeBlock(out_c, k_sz=k_sz)

    def forward(self, x, skip):
        up = self.up_layer(x)
        if self.conv_bridge:
            out = torch.cat([up, self.conv_bridge_layer(skip)], dim=1)
        else:
            out = torch.cat([up, skip], dim=1)
        out = self.conv_layer(out)
        return out


class LittleUNet(torch.nn.Module):
    """Little U-Net model."""

    def __init__(
        self,
        in_c,
        n_classes,
        layers,
        k_sz=3,
        up_mode="transp_conv",
        conv_bridge=True,
        shortcut=True,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.first = ConvBlock(
            in_c=in_c, out_c=layers[0], k_sz=k_sz, shortcut=shortcut, pool=False
        )

        self.down_path = torch.nn.ModuleList()
        for i in range(len(layers) - 1):
            block = ConvBlock(
                in_c=layers[i],
                out_c=layers[i + 1],
                k_sz=k_sz,
                shortcut=shortcut,
                pool=True,
            )
            self.down_path.append(block)

        self.up_path = torch.nn.ModuleList()
        reversed_layers = list(reversed(layers))
        for i in range(len(layers) - 1):
            block = UpConvBlock(
                in_c=reversed_layers[i],
                out_c=reversed_layers[i + 1],
                k_sz=k_sz,
                up_mode=up_mode,
                conv_bridge=conv_bridge,
                shortcut=shortcut,
            )
            self.up_path.append(block)

        # init, shamelessly lifted from torchvision/models/resnet.py
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

        self.final = torch.nn.Conv2d(layers[0], n_classes, kernel_size=1)

    def forward(self, x):
        x = self.first(x)
        down_activations = []
        for i, down in enumerate(self.down_path):
            down_activations.append(x)
            x = down(x)
        down_activations.reverse()
        for i, up in enumerate(self.up_path):
            x = up(x, down_activations[i])
        return self.final(x)


class LittleWNet(torch.nn.Module):
    """Little W-Net model, concatenating two Little U-Net models."""

    def __init__(
        self,
        n_classes=1,
        in_c=3,
        layers=(8, 16, 32),
        conv_bridge=True,
        shortcut=True,
        mode="train",
    ):
        super().__init__()
        self.unet1 = LittleUNet(
            in_c=in_c,
            n_classes=n_classes,
            layers=layers,
            conv_bridge=conv_bridge,
            shortcut=shortcut,
        )
        self.unet2 = LittleUNet(
            in_c=in_c + n_classes,
            n_classes=n_classes,
            layers=layers,
            conv_bridge=conv_bridge,
            shortcut=shortcut,
        )
        self.n_classes = n_classes
        self.mode = mode

    def forward(self, x):
        x1 = self.unet1(x)
        x2 = self.unet2(torch.cat([x, x1], dim=1))
        if self.mode != "train":
            return x2
        return x1, x2


def lunet(input_channels=3, output_classes=1):
    """Builds Little U-Net segmentation network (uninitialized)

    Parameters
    ----------

    input_channels : :py:class:`int`, Optional
        Number of input channels the network should operate with

    output_classes : :py:class:`int`, Optional
        Number of output classes


    Returns
    -------

    module : :py:class:`torch.nn.Module`
        Network model for Little U-Net
    """

    return LittleUNet(
        in_c=input_channels,
        n_classes=output_classes,
        layers=[8, 16, 32],
        conv_bridge=True,
        shortcut=True,
    )


def lwnet(input_channels=3, output_classes=1):
    """Builds Little W-Net segmentation network (uninitialized)

    Parameters
    ----------

    input_channels : :py:class:`int`, Optional
        Number of input channels the network should operate with

    output_classes : :py:class:`int`, Optional
        Number of output classes


    Returns
    -------

    module : :py:class:`torch.nn.Module`
        Network model for Little W-Net
    """

    return LittleWNet(
        in_c=input_channels,
        n_classes=output_classes,
        layers=[8, 16, 32],
        conv_bridge=True,
        shortcut=True,
    )
