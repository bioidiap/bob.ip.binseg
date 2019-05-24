#!/usr/bin/env python
# -*- coding: utf-8 -*-

from bob.db.drive import Database as DRIVE
from bob.db.hrf import Database as HRF
from bob.ip.binseg.data.transforms import *
from bob.ip.binseg.data.binsegdataset import BinSegDataset, SSLBinSegDataset, UnLabeledBinSegDataset

#### Config ####

#### Unlabeled HRF TRAIN ####
unlabeled_transforms = Compose([  
                        Crop(0,108,2336,3296)
                        ,RandomHFlip()
                        ,RandomVFlip()
                        ,RandomRotation()
                        ,ColorJitter()
                        ,ToTensor()
                    ])

# bob.db.dataset init
sslbobdb = HRF(protocol = 'default')

# PyTorch dataset
unlabeled_dataset = UnLabeledBinSegDataset(sslbobdb, split='train', transform=unlabeled_transforms)


#### Labeled ####
labeled_transforms = Compose([  
                        Crop(75,10,416,544)
                        ,Pad((21,0,22,0))
                        ,Resize(2336)
                        ,RandomHFlip()
                        ,RandomVFlip()
                        ,RandomRotation()
                        ,ColorJitter()
                        ,ToTensor()
                    ])

# bob.db.dataset init
bobdb = DRIVE(protocol = 'default')

# PyTorch dataset
dataset = SSLBinSegDataset(bobdb, unlabeled_dataset, split='train', transform=labeled_transforms)
