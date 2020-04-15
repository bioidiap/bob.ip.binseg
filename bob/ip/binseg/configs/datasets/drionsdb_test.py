#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""DRIONS-DB (test set) for Optic Disc Segmentation

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
* Training samples: 50
* Split reference: [MANINIS-2016]_
"""

from bob.ip.binseg.data.transforms import Pad
_transforms = [Pad((4, 8, 4, 8))]

from bob.ip.binseg.data.utils import SampleList2TorchDataset
from bob.ip.binseg.data.drionsdb import dataset as drionsdb
dataset = SampleList2TorchDataset(drionsdb.subsets("default")["test"],
        transforms=_transforms)
