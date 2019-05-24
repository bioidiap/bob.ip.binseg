#!/usr/bin/env python
# -*- coding: utf-8 -*-

from bob.db.drive import Database as DRIVE
from bob.db.iostar import Database as IOSTAR
from bob.ip.binseg.data.transforms import *
from bob.ip.binseg.data.binsegdataset import BinSegDataset, SSLBinSegDataset, UnLabeledBinSegDataset

#### Config ####

#### Unlabeled IOSTAR TRAIN ####
unlabeled_transforms = Compose([  
                        RandomHFlip()
                        ,RandomVFlip()
                        ,RandomRotation()
                        ,ColorJitter()
                        ,ToTensor()
                    ])

# bob.db.dataset init
sslbobdb = IOSTAR(protocol = 'default_vessel')

# PyTorch dataset
unlabeled_dataset = UnLabeledBinSegDataset(sslbobdb, split='train', transform=unlabeled_transforms)


#### Labeled ####
labeled_transforms = Compose([
                        CenterCrop((540,540))  
                        ,Resize(1024)
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
