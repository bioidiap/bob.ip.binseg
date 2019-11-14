#!/usr/bin/env python
# -*- coding: utf-8 -*-

from bob.ip.binseg.data.transforms import *
from bob.ip.binseg.data.imagefolderinference import ImageFolderInference

#### Config ####

# add your transforms below
transforms = Compose([
                        ToRGB(),
                        CenterCrop((544,544))
                        ,ToTensor()
                    ])

# PyTorch dataset
path = '/path/to/folder/containing/images'
dataset = ImageFolderInference(path,transform=transforms)
