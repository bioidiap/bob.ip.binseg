#!/usr/bin/env python
# -*- coding: utf-8 -*-

from bob.db.stare import Database as STARE
from bob.ip.binseg.data.transforms import *
from bob.ip.binseg.data.binsegdataset import BinSegDataset

#### Config ####

transforms = Compose([  
                        Pad((2,1,2,2))
                        ,ToTensor()
                    ])

# bob.db.dataset init
bobdb = STARE(protocol = 'default')

# PyTorch dataset
dataset = BinSegDataset(bobdb, split='test', transform=transforms)