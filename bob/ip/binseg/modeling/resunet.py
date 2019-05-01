#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
from collections import OrderedDict
from bob.ip.binseg.modeling.make_layers  import conv_with_kaiming_uniform, convtrans_with_kaiming_uniform, PixelShuffle_ICNR, UnetBlock
from bob.ip.binseg.modeling.backbones.resnet import resnet50



class ResUNet(nn.Module):
    """
    UNet head module for ResNet backbones
    
    Parameters
    ----------
    in_channels_list : list
                        number of channels for each feature map that is returned from backbone
    """
    def __init__(self, in_channels_list=None, pixel_shuffle=False):
        super(ResUNet, self).__init__()
        # number of channels
        c_decode1, c_decode2, c_decode3, c_decode4, c_decode5 = in_channels_list
        # number of channels for last upsampling operation
        c_decode0 = (c_decode1 + c_decode2//2)//2

        # build layers
        self.decode4 = UnetBlock(c_decode5, c_decode4, pixel_shuffle)
        self.decode3 = UnetBlock(c_decode4, c_decode3, pixel_shuffle)
        self.decode2 = UnetBlock(c_decode3, c_decode2, pixel_shuffle)
        self.decode1 = UnetBlock(c_decode2, c_decode1, pixel_shuffle)
        if pixel_shuffle:
            self.decode0 = PixelShuffle_ICNR(c_decode0, c_decode0)
        else:
            self.decode0 = convtrans_with_kaiming_uniform(c_decode0, c_decode0, 2, 2)
        self.final = conv_with_kaiming_uniform(c_decode0, 1, 1)

    def forward(self,x):
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
        decode0 = self.decode0(decode1)
        out = self.final(decode0)
        return out

def build_res50unet():
    """ 
    Adds backbone and head together

    Returns
    -------
    model : :py:class:torch.nn.Module
    """
    backbone = resnet50(pretrained=False, return_features = [2, 4, 5, 6, 7])
    unet_head  = ResUNet([64, 256, 512, 1024, 2048],pixel_shuffle=False)
    model = nn.Sequential(OrderedDict([("backbone", backbone), ("head", unet_head)]))
    model.name = "ResUNet"
    return model