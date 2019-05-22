#!/usr/bin/env python
# -*- coding: utf-8 -*-

from bob.db.drionsdb import Database as DRIONS
from bob.ip.binseg.data.transforms import *
from bob.ip.binseg.data.binsegdataset import BinSegDataset

#### Config ####

transforms = Compose([  
                        Pad((4,8,4,8))
                        ,ToTensor()
                    ])

# bob.db.dataset init
bobdb = DRIONS(protocol = 'default')

# PyTorch dataset
dataset = BinSegDataset(bobdb, split='test', transform=transforms)