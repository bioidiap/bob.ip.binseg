#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torchvision.transforms.functional as VF
import random
from PIL import Image

# Compose 

class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

# Preprocessing

class CenterCrop:
    """
    Crops the given PIL images the center.
    
    Attributes
    ----------
    size: (sequence or int)
        Desired output size of the crop. If size is an int instead of sequence like (h, w), a square crop (size, size) is made.
    
    """
    def __init__(self, size):
        self.size = size
        
    def __call__(self, *args):
        return [VF.center_crop(img, self.size) for img in args]


class Crop:
    """
    Crop the given PIL Image ground_truth at the given coordinates.
    Attributes
    ----------
        img (PIL Image): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
    """
    def __init__(self, i, j, h, w):
        self.i = i
        self.j = j
        self.h = h 
        self.w = w 
    
    def __call__(self, *args):
        return [img.crop((self.j, self.i, self.j + self.w, self.i + self.h)) for img in args]

class Pad:
    """
    Attributes
    ----------
    
    padding : int or tuple 
        Padding on each border. If a single int is provided this is used to pad all borders. 
        If tuple of length 2 is provided this is the padding on left/right and top/bottom respectively.
        If a tuple of length 4 is provided this is the padding for the left, top, right and bottom borders respectively.
    
    fill : int
        Pixel fill value for constant fill. Default is 0. If a tuple of length 3, it is used to fill R, G, B channels 
        respectively. This value is only used when the padding_mode is constant
        
    """
    def __init__(self, padding, fill=0):
        self.padding = padding
        self.fill = fill
        
    def __call__(self, *args):
        return [VF.pad(img, self.padding, self.fill, padding_mode='constant') for img in args]
    
class ToTensor:
    def __call__(self, *args):
        return [VF.to_tensor(img) for img in args]

        
# Augmentations

class RandomHFlip:
    """
    Flips the given PIL image and ground truth horizontally
    Attributes
    ----------
    
    prob : float 
        probability at which imgage is flipped. Default: 0.5
    """
    def __init__(self, prob = 0.5):
        self.prob = prob
        
    def __call__(self, *args):
        if random.random() < self.prob:
            return [VF.hflip(img) for img in args]
            
        else:
            return args
    
    
class RandomVFlip:
    """
    Flips the given PIL image and ground truth vertically
    Attributes
    ----------
    
    prob : float 
        probability at which imgage is flipped. Default: 0.5
    """
    def __init__(self, prob = 0.5):
        self.prob = prob
        
    def __call__(self, *args):
        if random.random() < self.prob:
            return [VF.vflip(img) for img in args]
            
        else:
            return args
    

class RandomRotation:
    """
    Rotates the given PIL image and ground truth vertically
    Attributes
    ----------
    
    prob : float 
        probability at which imgage is rotated. Default: 0.5
    degree_range : tuple
        range of degrees in which image and ground truth are rotated. Default: (-15, +15) 
    """
    def __init__(self, degree_range = (-15, +15), prob = 0.5):
        self.prob = prob
        self.degree_range = degree_range
        
    def __call__(self, *args):
        if random.random() < self.prob:
            degree = random.randint(*self.degree_range)
            return [VF.rotate(img, degree, resample = Image.BILINEAR) for img in args]
        else:
            return args