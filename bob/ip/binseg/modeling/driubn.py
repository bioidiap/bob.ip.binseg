#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
from collections import OrderedDict
from bob.ip.binseg.modeling.backbones.vgg import vgg16_bn
from bob.ip.binseg.modeling.make_layers import conv_with_kaiming_uniform,convtrans_with_kaiming_uniform, UpsampleCropBlock

class ConcatFuseBlock(nn.Module):
    """ 
    Takes in four feature maps with 16 channels each, concatenates them 
    and applies a 1x1 convolution with 1 output channel. 
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            conv_with_kaiming_uniform(4*16,1,1,1,0)
            ,nn.BatchNorm2d(1)
        )
    def forward(self,x1,x2,x3,x4):
        x_cat = torch.cat([x1,x2,x3,x4],dim=1)
        x = self.conv(x_cat)
        return x 
            
class DRIU(nn.Module):
    """
    DRIU head module
    Based on paper by `Maninis et al. (2016)`_ 
    Parameters
    ----------
    in_channels_list : list
        number of channels for each feature map that is returned from backbone
    """
    def __init__(self, in_channels_list=None):
        super(DRIU, self).__init__()
        in_conv_1_2_16, in_upsample2, in_upsample_4, in_upsample_8 = in_channels_list

        self.conv1_2_16 = nn.Conv2d(in_conv_1_2_16, 16, 3, 1, 1)
        # Upsample layers
        self.upsample2 = UpsampleCropBlock(in_upsample2, 16, 4, 2, 0)
        self.upsample4 = UpsampleCropBlock(in_upsample_4, 16, 8, 4, 0)
        self.upsample8 = UpsampleCropBlock(in_upsample_8, 16, 16, 8, 0)
        
        # Concat and Fuse
        self.concatfuse = ConcatFuseBlock()

    def forward(self,x):
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
        upsample2 = self.upsample2(x[2], hw) # side-multi2-up
        upsample4 = self.upsample4(x[3], hw) # side-multi3-up
        upsample8 = self.upsample8(x[4], hw) # side-multi4-up
        out = self.concatfuse(conv1_2_16, upsample2, upsample4, upsample8)
        return out

def build_driu():
    """ 
    Adds backbone and head together

    Returns
    -------
    :py:class:torch.nn.Module
    """
    backbone = vgg16_bn(pretrained=False, return_features = [5, 12, 19, 29])
    driu_head = DRIU([64, 128, 256, 512])

    model = nn.Sequential(OrderedDict([("backbone", backbone), ("head", driu_head)]))
    model.name = "DRIUBN"
    return model