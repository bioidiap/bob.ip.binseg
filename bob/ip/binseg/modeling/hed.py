#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
from collections import OrderedDict
from bob.ip.binseg.modeling.backbones.vgg import vgg16
from bob.ip.binseg.modeling.make_layers import conv_with_kaiming_uniform, convtrans_with_kaiming_uniform, UpsampleCropBlock

class ConcatFuseBlock(nn.Module):
    """ 
    Takes in five feature maps with one channel each, concatenates thems 
    and applies a 1x1 convolution with 1 output channel. 
    """
    def __init__(self):
        super().__init__()
        self.conv = conv_with_kaiming_uniform(5,1,1,1,0)
    
    def forward(self,x1,x2,x3,x4,x5):
        x_cat = torch.cat([x1,x2,x3,x4,x5],dim=1)
        x = self.conv(x_cat)
        return x 
            
class HED(nn.Module):
    """
    HED head module
    
    Parameters
    ----------
    in_channels_list : list
        number of channels for each feature map that is returned from backbone
    """
    def __init__(self, in_channels_list=None):
        super(HED, self).__init__()
        in_conv_1_2_16, in_upsample2, in_upsample_4, in_upsample_8, in_upsample_16 = in_channels_list
        
        self.conv1_2_16 = nn.Conv2d(in_conv_1_2_16,1,3,1,1)
        # Upsample
        self.upsample2 = UpsampleCropBlock(in_upsample2,1,4,2,0)
        self.upsample4 = UpsampleCropBlock(in_upsample_4,1,8,4,0)
        self.upsample8 = UpsampleCropBlock(in_upsample_8,1,16,8,0)
        self.upsample16 = UpsampleCropBlock(in_upsample_16,1,32,16,0)
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
        conv1_2_16 = self.conv1_2_16(x[1])  
        upsample2 = self.upsample2(x[2],hw)
        upsample4 = self.upsample4(x[3],hw)
        upsample8 = self.upsample8(x[4],hw)
        upsample16 = self.upsample16(x[5],hw) 
        concatfuse = self.concatfuse(conv1_2_16,upsample2,upsample4,upsample8,upsample16)
        
        out = [upsample2,upsample4,upsample8,upsample16,concatfuse]
        return out

def build_hed():
    """ 
    Adds backbone and head together

    Returns
    -------
    :py:class:torch.nn.Module
    """
    backbone = vgg16(pretrained=False, return_features = [3, 8, 14, 22, 29])
    hed_head = HED([64, 128, 256, 512, 512])

    model = nn.Sequential(OrderedDict([("backbone", backbone), ("head", hed_head)]))
    model.name = "HED"
    return model