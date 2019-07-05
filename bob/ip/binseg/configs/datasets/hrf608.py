#!/usr/bin/env python
# -*- coding: utf-8 -*-

from bob.db.hrf import Database as HRF
from bob.ip.binseg.data.transforms import *
from bob.ip.binseg.data.binsegdataset import BinSegDataset

#### Config ####

transforms = Compose([  
                        Pad((0,345,0,345))
                        ,Resize(608)
                        ,RandomRotation()
                        ,RandomHFlip()
                        ,RandomVFlip()
                        ,ColorJitter()
                        ,ToTensor()
                    ])

# bob.db.dataset init
bobdb = HRF(protocol = 'default')

# PyTorch dataset
dataset = BinSegDataset(bobdb, split='train', transform=transforms)
