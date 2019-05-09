#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

# https://github.com/laibe/M2U-Net

from collections import OrderedDict
import torch
from torch import nn
from bob.ip.binseg.modeling.backbones.mobilenetv2 import MobileNetV2, InvertedResidual

class DecoderBlock(nn.Module):
    """
    Decoder block: upsample and concatenate with features maps from the encoder part
    """
    def __init__(self,up_in_c,x_in_c,upsamplemode='bilinear',expand_ratio=0.15):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2,mode=upsamplemode,align_corners=False) # H, W -> 2H, 2W
        self.ir1 = InvertedResidual(up_in_c+x_in_c,(x_in_c + up_in_c) // 2,stride=1,expand_ratio=expand_ratio)

    def forward(self,up_in,x_in):
        up_out = self.upsample(up_in)
        cat_x = torch.cat([up_out, x_in] , dim=1)
        x = self.ir1(cat_x)
        return x
    
class LastDecoderBlock(nn.Module):
    def __init__(self,x_in_c,upsamplemode='bilinear',expand_ratio=0.15):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2,mode=upsamplemode,align_corners=False) # H, W -> 2H, 2W
        self.ir1 = InvertedResidual(x_in_c,1,stride=1,expand_ratio=expand_ratio)

    def forward(self,up_in,x_in):
        up_out = self.upsample(up_in)
        cat_x = torch.cat([up_out, x_in] , dim=1)
        x = self.ir1(cat_x)
        return x



class M2U(nn.Module):
    """
    M2U-Net head module
    
    Parameters
    ----------
    in_channels_list : list
        number of channels for each feature map that is returned from backbone
    """
    def __init__(self, in_channels_list=None,upsamplemode='bilinear',expand_ratio=0.15):
        super(M2U, self).__init__()

        # Decoder
        self.decode4 = DecoderBlock(96,32,upsamplemode,expand_ratio)
        self.decode3 = DecoderBlock(64,24,upsamplemode,expand_ratio)
        self.decode2 = DecoderBlock(44,16,upsamplemode,expand_ratio)
        self.decode1 = LastDecoderBlock(33,upsamplemode,expand_ratio)
        
        # initilaize weights 
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
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
        decode4 = self.decode4(x[5],x[4])    # 96, 32
        decode3 = self.decode3(decode4,x[3]) # 64, 24
        decode2 = self.decode2(decode3,x[2]) # 44, 16
        decode1 = self.decode1(decode2,x[1]) # 30, 3
        
        return decode1

def build_m2unet():
    """ 
    Adds backbone and head together

    Returns
    -------
    :py:class:torch.nn.Module
    """
    backbone = MobileNetV2(return_features = [1, 3, 6, 13], m2u=True)
    m2u_head = M2U(in_channels_list=[16, 24, 32, 96])

    model = nn.Sequential(OrderedDict([("backbone", backbone), ("head", m2u_head)]))
    model.name = "M2UNet"
    return model