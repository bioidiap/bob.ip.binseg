#!/usr/bin/env python
# -*- coding: utf-8 -*-

from bob.db.stare import Database as STARE
from bob.ip.binseg.data.transforms import *
from bob.ip.binseg.data.binsegdataset import BinSegDataset
import torch
#### Config ####

transforms = Compose([  
                        RandomRotation()
                        ,Crop(50,0,500,705)
                        ,Resize(1168)
                        ,Pad((1,0,1,0))
                        ,RandomHFlip()
                        ,RandomVFlip()
                        ,ColorJitter()
                        ,ToTensor()
                    ])

# bob.db.dataset init
bobdb = STARE(protocol = 'default')

# PyTorch dataset
train = BinSegDataset(bobdb, split='train', transform=transforms)
test = BinSegDataset(bobdb, split='test', transform=transforms)

dataset = torch.utils.data.ConcatDataset([train,test])