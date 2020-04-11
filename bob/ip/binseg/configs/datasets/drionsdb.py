#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""DRIONS-DB (training set) for Optic Disc Segmentation

The dataset originates from data collected from 55 patients with glaucoma
(23.1%) and eye hypertension (76.9%), and random selected from an eye fundus
image base belonging to the Ophthalmology Service at Miguel Servet Hospital,
Saragossa (Spain).  It contains 110 eye fundus images with a resolution of 600
x 400. Two sets of ground-truth optic disc annotations are available. The first
set is commonly used for training and testing. The second set acts as a “human”
baseline.

* Reference: [DRIONSDB-2008]_
* Original resolution (height x width): 400 x 600
* Configuration resolution: 416 x 608 (after padding)
* Training samples: 60
* Split reference: [MANINIS-2016]_
"""

from bob.ip.binseg.data.transforms import Pad
from bob.ip.binseg.configs.datasets.utils import DATA_AUGMENTATION as _DA
_transforms = [Pad((4, 8, 4, 8))] + _DA

from bob.db.drionsdb import Database as DRIONS
bobdb = DRIONS(protocol="default")

from bob.ip.binseg.data.binsegdataset import BinSegDataset
dataset = BinSegDataset(bobdb, split="train", transforms=_transforms)
