#!/usr/bin/env python
# -*- coding: utf-8 -*-
from bob.db.drive import Database as DRIVE
from bob.db.stare import Database as STARE
from bob.db.chasedb1 import Database as CHASEDB1
from bob.db.iostar import Database as IOSTAR
from bob.db.hrf import Database as HRF
from bob.ip.binseg.data.transforms import *
from bob.ip.binseg.data.binsegdataset import BinSegDataset
import torch

# Target size: 544x544 (DRIVE)

defaulttransforms = [ToTensor()]


# CHASE_DB1 
transforms_chase = Compose([      
                        Resize(544)
                        ,Crop(0,12,544,544)
                        ,*defaulttransforms
                    ])

# bob.db.dataset init
bobdb_chase = CHASEDB1(protocol = 'default')

# PyTorch dataset
torch_chase = BinSegDataset(bobdb_chase, split='test', transform=transforms_chase)


# IOSTAR VESSEL
transforms_iostar = Compose([  
                        Resize(544)
                        ,*defaulttransforms
                    ])

# bob.db.dataset init
bobdb_iostar = IOSTAR(protocol='default_vessel')

# PyTorch dataset
torch_iostar = BinSegDataset(bobdb_iostar, split='test', transform=transforms_iostar)

# STARE
transforms = Compose([  
                        Resize(471)
                        ,Pad((0,37,0,36))
                        ,*defaulttransforms
                    ])

# bob.db.dataset init
bobdb_stare = STARE(protocol = 'default')

# PyTorch dataset
torch_stare = BinSegDataset(bobdb_stare, split='test', transform=transforms)


# HRF
transforms_hrf = Compose([  
                        Resize((363))
                        ,Pad((0,90,0,91))
                        ,*defaulttransforms
                    ])

# bob.db.dataset init
bobdb_hrf = HRF(protocol = 'default')

# PyTorch dataset
torch_hrf = BinSegDataset(bobdb_hrf, split='test', transform=transforms_hrf)



# Merge
dataset = torch.utils.data.ConcatDataset([torch_stare, torch_chase, torch_iostar, torch_hrf])