#!/usr/bin/env python
# -*- coding: utf-8 -*-

from bob.db.stare import Database as STARE
from bob.db.refuge import Database as REFUGE
from bob.db.hrf import Database as HRF
from bob.ip.binseg.data.transforms import *
from bob.ip.binseg.data.binsegdataset import BinSegDataset, SSLBinSegDataset, UnLabeledBinSegDataset
import torch 
#### Config ####

#### Unlabeled HRF TRAIN ####
unlabeled_transforms = Compose([  
                        RandomRotation()
                        ,Crop(0,108,2336,3296)
                        ,Resize(1168)
                        ,RandomHFlip()
                        ,RandomVFlip()
                        ,ColorJitter()
                        ,ToTensor()
                    ])

#### Unlabeled REFUGE Test ####

unlabeled_transforms_refuge = Compose([  
                        RandomRotation()
                        ,Crop(220,11,1150,1623)
                        ,Resize(1168)
                        ,RandomHFlip()
                        ,RandomVFlip()
                        ,ColorJitter()
                        ,ToTensor()
                    ])

## bob.db.dataset init
# hrf
hrfbobdb = HRF(protocol = 'default')
# refuge
refugebobdb = REFUGE()


# PyTorch dataset
unlabeled_dataset_1 = UnLabeledBinSegDataset(hrfbobdb, split='train', transform=unlabeled_transforms)

unlabeled_dataset_2 = UnLabeledBinSegDataset(refugebobdb, split='test', transform=unlabeled_transforms_refuge)

# Compose
unlabeled_dataset = torch.utils.data.ConcatDataset([unlabeled_dataset_1,unlabeled_dataset_2])


#### Labeled ####
labeled_transforms = Compose([  
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
labeled_dataset = BinSegDataset(bobdb, split='train', transform=labeled_transforms)

# SSL Dataset

dataset = SSLBinSegDataset(labeled_dataset, unlabeled_dataset)