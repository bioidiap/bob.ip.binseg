#!/usr/bin/env python
# -*- coding: utf-8 -*-

from bob.ip.binseg.data.transforms import *
from bob.ip.binseg.data.imagefolder import ImageFolder

#### Config ####

# add your transforms below
transforms = Compose([  
                        CenterCrop((544,544))
                        ,ToTensor()
                    ])

# PyTorch dataset
path = '/path/to/testdataset'
dataset = ImageFolder(path,transform=transforms)
