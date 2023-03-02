# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import torch
import torch.nn

from torch.nn import Conv2d, ConvTranspose2d


def conv_with_kaiming_uniform(
    in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1
):
    conv = Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=True,
    )
    # Caffe2 implementation uses XavierFill, which in fact
    # corresponds to kaiming_uniform_ in PyTorch
    torch.nn.init.kaiming_uniform_(conv.weight, a=1)
    torch.nn.init.constant_(conv.bias, 0)
    return conv


def convtrans_with_kaiming_uniform(
    in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1
):
    conv = ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=True,
    )
    # Caffe2 implementation uses XavierFill, which in fact
    # corresponds to kaiming_uniform_ in PyTorch
    torch.nn.init.kaiming_uniform_(conv.weight, a=1)
    torch.nn.init.constant_(conv.bias, 0)
    return conv


class UpsampleCropBlock(torch.nn.Module):
    """Combines Conv2d, ConvTransposed2d and Cropping. Simulates the caffe2
    crop layer in the forward function.

    Used for DRIU and HED.

    Parameters
    ----------

    in_channels : int
        number of channels of intermediate layer
    out_channels : int
        number of output channels
    up_kernel_size : int
        kernel size for transposed convolution
    up_stride : int
        stride for transposed convolution
    up_padding : int
        padding for transposed convolution
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        up_kernel_size,
        up_stride,
        up_padding,
        pixelshuffle=False,
    ):
        super().__init__()
        # NOTE: Kaiming init, replace with torch.nn.Conv2d and torch.nn.ConvTranspose2d to get original DRIU impl.
        self.conv = conv_with_kaiming_uniform(
            in_channels, out_channels, 3, 1, 1
        )
        if pixelshuffle:
            self.upconv = PixelShuffle_ICNR(
                out_channels, out_channels, scale=up_stride
            )
        else:
            self.upconv = convtrans_with_kaiming_uniform(
                out_channels,
                out_channels,
                up_kernel_size,
                up_stride,
                up_padding,
            )

    def forward(self, x, input_res):
        """Forward pass of UpsampleBlock.

        Upsampled feature maps are cropped to the resolution of the input
        image.

        Parameters
        ----------

        x : tuple
            input channels

        input_res : tuple
            Resolution of the input image format ``(height, width)``
        """

        img_h = input_res[0]
        img_w = input_res[1]
        x = self.conv(x)
        x = self.upconv(x)
        # determine center crop
        # height
        up_h = x.shape[2]
        h_crop = up_h - img_h
        h_s = h_crop // 2
        h_e = up_h - (h_crop - h_s)
        # width
        up_w = x.shape[3]
        w_crop = up_w - img_w
        w_s = w_crop // 2
        w_e = up_w - (w_crop - w_s)
        # perform crop
        # needs explicit ranges for onnx export
        x = x[:, :, h_s:h_e, w_s:w_e]  # crop to input size

        return x


def ifnone(a, b):
    "``a`` if ``a`` is not None, otherwise ``b``."
    return b if a is None else a


def icnr(x, scale=2, init=torch.nn.init.kaiming_normal_):
    """https://docs.fast.ai/layers.html#PixelShuffle_ICNR.

    ICNR init of ``x``, with ``scale`` and ``init`` function.
    """

    ni, nf, h, w = x.shape
    ni2 = int(ni / (scale**2))
    k = init(torch.zeros([ni2, nf, h, w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale**2)
    k = k.contiguous().view([nf, ni, h, w]).transpose(0, 1)
    x.data.copy_(k)


class PixelShuffle_ICNR(torch.nn.Module):
    """https://docs.fast.ai/layers.html#PixelShuffle_ICNR.

    Upsample by ``scale`` from ``ni`` filters to ``nf`` (default
    ``ni``), using ``torch.nn.PixelShuffle``, ``icnr`` init, and
    ``weight_norm``.
    """

    def __init__(self, ni: int, nf: int = None, scale: int = 2):
        super().__init__()
        nf = ifnone(nf, ni)
        self.conv = conv_with_kaiming_uniform(ni, nf * (scale**2), 1)
        icnr(self.conv.weight)
        self.shuf = torch.nn.PixelShuffle(scale)
        # Blurring over (h*w) kernel
        # "Super-Resolution using Convolutional Neural Networks without Any Checkerboard Artifacts"
        # - https://arxiv.org/abs/1806.02658
        self.pad = torch.nn.ReplicationPad2d((1, 0, 1, 0))
        self.blur = torch.nn.AvgPool2d(2, stride=1)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.shuf(self.relu(self.conv(x)))
        x = self.blur(self.pad(x))
        return x


class UnetBlock(torch.nn.Module):
    def __init__(
        self, up_in_c, x_in_c, pixel_shuffle=False, middle_block=False
    ):
        super().__init__()

        # middle block for VGG based U-Net
        if middle_block:
            up_out_c = up_in_c
        else:
            up_out_c = up_in_c // 2
        cat_channels = x_in_c + up_out_c
        inner_channels = cat_channels // 2

        if pixel_shuffle:
            self.upsample = PixelShuffle_ICNR(up_in_c, up_out_c)
        else:
            self.upsample = convtrans_with_kaiming_uniform(
                up_in_c, up_out_c, 2, 2
            )
        self.convtrans1 = convtrans_with_kaiming_uniform(
            cat_channels, inner_channels, 3, 1, 1
        )
        self.convtrans2 = convtrans_with_kaiming_uniform(
            inner_channels, inner_channels, 3, 1, 1
        )
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, up_in, x_in):
        up_out = self.upsample(up_in)
        cat_x = torch.cat([up_out, x_in], dim=1)
        x = self.relu(self.convtrans1(cat_x))
        x = self.relu(self.convtrans2(x))
        return x
