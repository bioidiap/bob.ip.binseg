#!/usr/bin/env python
# -*- coding: utf-8 -*-

from bob.db.stare import Database as STARE
from bob.ip.binseg.data.transforms import *
from bob.ip.binseg.data.binsegdataset import BinSegDataset

#### Config ####

transforms = Compose([  RandomRotation()
                        ,Resize(471)
                        ,Pad((0,37,0,36))
                        ,RandomHFlip()
                        ,RandomVFlip()
                        ,ColorJitter()
                        ,ToTensor()
                    ])

# bob.db.dataset init
bobdb = STARE(protocol = 'default')

# PyTorch dataset
dataset = BinSegDataset(bobdb, split='train', transform=transforms)