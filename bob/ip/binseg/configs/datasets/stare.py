#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""STARE (training set) for Vessel Segmentation

A subset of the original STARE dataset contains 20 annotated eye fundus images
with a resolution of 605 x 700 (height x width). Two sets of ground-truth
vessel annotations are available. The first set by Adam Hoover is commonly used
for training and testing. The second set by Valentina Kouznetsova acts as a
“human” baseline.

* Reference: [STARE-2000]_
* Original resolution (width x height): 700 x 605
* Configuration resolution: 704 x 608 (after padding)
* Training samples: 10
* Split reference: [MANINIS-2016]_
"""

from bob.ip.binseg.data.transforms import *
_transforms = [
        Pad((2, 1, 2, 2)),  #(left, top, right, bottom)
        RandomHFlip(),
        RandomVFlip(),
        RandomRotation(),
        ColorJitter(),
        ]

from bob.ip.binseg.data.utils import SampleList2TorchDataset
from bob.ip.binseg.data.stare import dataset as stare
dataset = SampleList2TorchDataset(stare.subsets("default")["train"],
        transforms=_transforms)
