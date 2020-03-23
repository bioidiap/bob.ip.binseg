#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""STARE (test set) for Vessel Segmentation

A subset of the original STARE dataset contains 20 annotated eye fundus images
with a resolution of 605 x 700 (height x width). Two sets of ground-truth
vessel annotations are available. The first set by Adam Hoover is commonly used
for training and testing. The second set by Valentina Kouznetsova acts as a
“human” baseline.

* Reference: [STARE-2000]_
* Original resolution (height x width): 605 x 700
* Configuration resolution: 608 x 704 (after padding)
* Test samples: 10
* Split reference: [MANINIS-2016]_
"""

from bob.db.stare import Database as STARE
from bob.ip.binseg.data.transforms import *
from bob.ip.binseg.data.binsegdataset import BinSegDataset

#### Config ####

transforms = Compose([Pad((2, 1, 2, 2)), ToTensor()])

# bob.db.dataset init
bobdb = STARE(protocol="default")

# PyTorch dataset
dataset = BinSegDataset(bobdb, split="test", transform=transforms)
