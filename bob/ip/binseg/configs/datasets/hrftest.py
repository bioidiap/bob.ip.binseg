#!/usr/bin/env python
# -*- coding: utf-8 -*-

from bob.db.hrf import Database as HRF
from bob.ip.binseg.data.transforms import *
from bob.ip.binseg.data.binsegdataset import BinSegDataset

#### Config ####

transforms = Compose([  
                        Crop(0,108,2336,3296)
                        ,ToTensor()
                    ])

# bob.db.dataset init
bobdb = HRF(protocol = 'default')

# PyTorch dataset
dataset = BinSegDataset(bobdb, split='test', transform=transforms)