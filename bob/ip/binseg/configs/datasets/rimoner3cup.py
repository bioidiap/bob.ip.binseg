#!/usr/bin/env python
# -*- coding: utf-8 -*-

from bob.db.rimoner3 import Database as RIMONER3
from bob.ip.binseg.data.transforms import *
from bob.ip.binseg.data.binsegdataset import BinSegDataset

#### Config ####

transforms = Compose([  
                        Pad((8,8,8,8))
                        ,RandomHFlip()
                        ,RandomVFlip()
                        ,RandomRotation()
                        ,ColorJitter()
                        ,ToTensor()
                    ])

# bob.db.dataset init
bobdb = RIMONER3(protocol = 'default_cup')

# PyTorch dataset
dataset = BinSegDataset(bobdb, split='train', transform=transforms)