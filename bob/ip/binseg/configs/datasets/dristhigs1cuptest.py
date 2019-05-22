#!/usr/bin/env python
# -*- coding: utf-8 -*-
from bob.db.drishtigs1 import Database as DRISHTI
from bob.ip.binseg.data.transforms import *
from bob.ip.binseg.data.binsegdataset import BinSegDataset

#### Config ####

transforms = Compose([  
                        CenterCrop((1760,2048))
                        ,ToTensor()
                    ])

# bob.db.dataset init
bobdb = DRISHTI(protocol = 'default_cup')

# PyTorch dataset
dataset = BinSegDataset(bobdb, split='test', transform=transforms)