#!/usr/bin/env python
# -*- coding: utf-8 -*-

from bob.db.iostar import Database as IOSTAR
from bob.ip.binseg.data.transforms import *
from bob.ip.binseg.data.binsegdataset import BinSegDataset

#### Config ####

transforms = Compose([  
                        RandomRotation()
                        ,Crop(144,0,768,1024)
                        ,Pad((30,0,30,0))
                        ,Resize(1168)
                        ,RandomHFlip()
                        ,RandomVFlip()
                        ,ColorJitter()
                        ,ToTensor()
                    ])

# bob.db.dataset init
bobdb = IOSTAR(protocol='default_vessel')

# PyTorch dataset
dataset = BinSegDataset(bobdb, split='train', transform=transforms)