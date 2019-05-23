#!/usr/bin/env python
# -*- coding: utf-8 -*-

from bob.db.drive import Database as DRIVE
from bob.ip.binseg.data.transforms import *
from bob.ip.binseg.data.binsegdataset import BinSegDataset

#### Config ####

transforms = Compose([  
                        CenterCrop((540,540))
                        ,Resize(1024)
                        ,ToTensor()
                    ])

# bob.db.dataset init
bobdb = DRIVE(protocol = 'default')

# PyTorch dataset
dataset = BinSegDataset(bobdb, split='test', transform=transforms)