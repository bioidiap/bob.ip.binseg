#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn import Conv2d
from torch.nn import ConvTranspose2d

def conv_with_kaiming_uniform(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
    conv = Conv2d(
        in_channels, 
        out_channels, 
        kernel_size=kernel_size, 
        stride=stride, 
        padding=padding, 
        dilation=dilation, 
        bias= True
        )
        # Caffe2 implementation uses XavierFill, which in fact
        # corresponds to kaiming_uniform_ in PyTorch
    nn.init.kaiming_uniform_(conv.weight, a=1)
    nn.init.constant_(conv.bias, 0)
    return conv


def convtrans_with_kaiming_uniform(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
    conv = ConvTranspose2d(
        in_channels, 
        out_channels, 
        kernel_size=kernel_size, 
        stride=stride, 
        padding=padding, 
        dilation=dilation, 
        bias= True
        )
        # Caffe2 implementation uses XavierFill, which in fact
        # corresponds to kaiming_uniform_ in PyTorch
    nn.init.kaiming_uniform_(conv.weight, a=1)
    nn.init.constant_(conv.bias, 0)
    return conv


class UpsampleCropBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_kernel_size, up_stride, up_padding):
        """
        Combines Conv2d, ConvTransposed2d and Cropping. Simulates the caffe2 crop layer in the forward function.
        Used for DRIU and HED. 
        
        Attributes
        ----------
            in_channels : number of channels of intermediate layer
            out_channels : number of output channels
            up_kernel_size : kernel size for transposed convolution
            up_stride : stride for transposed convolution
            up_padding : padding for transposed convolution
        """
        super().__init__()
        # NOTE: Kaiming init, replace with nn.Conv2d and nn.ConvTranspose2d to get original DRIU impl.
        self.conv = conv_with_kaiming_uniform(in_channels, out_channels, 3, 1, 1)
        self.upconv = convtrans_with_kaiming_uniform(out_channels, out_channels, up_kernel_size, up_stride, up_padding)        
        
        
    def forward(self, x, input_res):
        """
        Forward pass of UpsampleBlock. Upsampled feature maps are cropped to the resolution of the input image.
        Attributes
        ----------
        x : input channels
        input_res : tuple (h,w)    
            Resolution of the input image
        """
        img_h = input_res[0]
        img_w = input_res[1]
        x = self.conv(x)
        x = self.upconv(x)
        # determine center crop
        # height
        up_h = x.shape[2]
        h_crop = up_h - img_h
        h_s = h_crop//2
        h_e = up_h - (h_crop - h_s)
        # width
        up_w = x.shape[3]
        w_crop = up_w-img_w
        w_s = w_crop//2
        w_e = up_w - (w_crop - w_s)
        # perform crop 
        # needs explicit ranges for onnx export 
        x = x[:,:,h_s:h_e,w_s:w_e] # crop to input size 
        
        return x