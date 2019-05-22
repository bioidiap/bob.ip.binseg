#!/usr/bin/env python
# -*- coding: utf-8 -*-

from bob.db.iostar import Database as IOSTAR
from bob.ip.binseg.data.transforms import *
from bob.ip.binseg.data.binsegdataset import BinSegDataset

#### Config ####

transforms = Compose([  
                        Resize(544)
                        ,ToTensor()
                    ])

# bob.db.dataset init
bobdb = IOSTAR(protocol='default_vessel')

# PyTorch dataset
dataset = BinSegDataset(bobdb, split='test', transform=transforms)