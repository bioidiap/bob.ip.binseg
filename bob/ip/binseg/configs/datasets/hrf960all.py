#!/usr/bin/env python
# -*- coding: utf-8 -*-

from bob.db.hrf import Database as HRF
from bob.ip.binseg.data.transforms import *
from bob.ip.binseg.data.binsegdataset import BinSegDataset
import torch

#### Config ####

transforms = Compose([  
                        Pad((0,584,0,584))                    
                        ,Resize((960))
                        ,RandomRotation()
                        ,RandomHFlip()
                        ,RandomVFlip()
                        ,ColorJitter()
                        ,ToTensor()
                    ])

# bob.db.dataset init
bobdb = HRF(protocol = 'default')

# PyTorch dataset
train = BinSegDataset(bobdb, split='train', transform=transforms)
test = BinSegDataset(bobdb, split='test', transform=transforms)

dataset = torch.utils.data.ConcatDataset([train,test])