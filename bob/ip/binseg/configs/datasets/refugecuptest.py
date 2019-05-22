#!/usr/bin/env python
# -*- coding: utf-8 -*-

from bob.db.refuge import Database as REFUGE
from bob.ip.binseg.data.transforms import *
from bob.ip.binseg.data.binsegdataset import BinSegDataset

#### Config ####

transforms = Compose([  
                        CenterCrop(1632)
                        ,ToTensor()
                    ])

# bob.db.dataset init
bobdb = REFUGE(protocol = 'default_cup')

# PyTorch dataset
dataset = BinSegDataset(bobdb, split='test', transform=transforms)