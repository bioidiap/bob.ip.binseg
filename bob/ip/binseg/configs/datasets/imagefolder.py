#!/usr/bin/env python
# -*- coding: utf-8 -*-

from bob.ip.binseg.data.transforms import *
from bob.ip.binseg.data.imagefolder import ImageFolder

#### Config ####

# add your transforms below
transforms = Compose([  
                        CenterCrop((544,544))
                        ,RandomHFlip()
                        ,RandomVFlip()
                        ,RandomRotation()
                        ,ColorJitter()
                        ,ToTensor()
                    ])

# PyTorch dataset
path = '/path/to/dataset'
dataset = ImageFolder(path,transform=transforms)
