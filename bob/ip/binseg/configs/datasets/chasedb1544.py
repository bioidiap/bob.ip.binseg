#!/usr/bin/env python
# -*- coding: utf-8 -*-

from bob.db.chasedb1 import Database as CHASEDB1
from bob.ip.binseg.data.transforms import *
from bob.ip.binseg.data.binsegdataset import BinSegDataset

#### Config ####

transforms = Compose([  
                        RandomRotation()
                        ,Resize(544)
                        ,Crop(0,12,544,544)
                        ,RandomHFlip()
                        ,RandomVFlip()
                        ,ColorJitter()
                        ,ToTensor()
                    ])

# bob.db.dataset init
bobdb = CHASEDB1(protocol = 'default')

# PyTorch dataset
dataset = BinSegDataset(bobdb, split='train', transform=transforms)