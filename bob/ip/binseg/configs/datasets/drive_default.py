#!/usr/bin/env python
# -*- coding: utf-8 -*-

from bob.ip.binseg.data.transforms import ToTensor
from bob.ip.binseg.data.binsegdataset import BinSegDataset
from torch.utils.data import DataLoader
from bob.db.drive import Database as DRIVE
import torch


#### Config ####

# bob.db.dataset init
bobdb = DRIVE()

# transforms 
transforms = ToTensor()
