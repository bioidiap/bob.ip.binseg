#!/usr/bin/env python
# -*- coding: utf-8 -*-

from bob.db.drive import Database as DRIVE
from bob.ip.binseg.data.transforms import *
from bob.ip.binseg.data.binsegdataset import BinSegDataset

#### Config ####

transforms = Compose([  
                        RandomRotation()
                        ,Crop(75,10,416,544)
                        ,Pad((21,0,22,0))
                        ,Resize(1168)
                        ,RandomHFlip()
                        ,RandomVFlip()
                        ,ColorJitter()
                        ,ToTensor()
                    ])

# bob.db.dataset init
bobdb = DRIVE(protocol = 'default')

# PyTorch dataset
dataset = BinSegDataset(bobdb, split='train', transform=transforms)