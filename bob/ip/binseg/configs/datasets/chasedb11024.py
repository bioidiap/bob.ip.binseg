#!/usr/bin/env python
# -*- coding: utf-8 -*-

from bob.db.chasedb1 import Database as CHASEDB1
from bob.ip.binseg.data.transforms import *
from bob.ip.binseg.data.binsegdataset import BinSegDataset

#### Config ####

transforms = Compose([  
                        RandomRotation()
                        ,Crop(0,18,960,960)
                        ,Resize(1024)
                        ,RandomHFlip()
                        ,RandomVFlip()
                        ,ColorJitter()
                        ,ToTensor()
                    ])

# bob.db.dataset init
bobdb = CHASEDB1(protocol = 'default')

# PyTorch dataset
dataset = BinSegDataset(bobdb, split='train', transform=transforms)