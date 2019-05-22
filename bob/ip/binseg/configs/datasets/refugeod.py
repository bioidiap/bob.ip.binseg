#!/usr/bin/env python
# -*- coding: utf-8 -*-

from bob.db.refuge import Database as REFUGE
from bob.ip.binseg.data.transforms import *
from bob.ip.binseg.data.binsegdataset import BinSegDataset

#### Config ####

transforms = Compose([  
                        Resize((1539))
                        ,Pad((21,46,22,47))
                        ,RandomHFlip()
                        ,RandomVFlip()
                        ,RandomRotation()
                        ,ColorJitter()
                        ,ToTensor()
                    ])

# bob.db.dataset init
bobdb = REFUGE(protocol = 'default_od')

# PyTorch dataset
dataset = BinSegDataset(bobdb, split='train', transform=transforms)