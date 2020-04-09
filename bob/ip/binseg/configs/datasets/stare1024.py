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
* Configuration resolution: 1024 x 1024
* Training samples: 10
* Split reference: [MANINIS-2016]_
"""

from bob.ip.binseg.data.transforms import *
_transforms = Compose(
    [
        RandomRotation(),
        Pad((0, 32, 0, 32)),
        Resize(1024),
        CenterCrop(1024),
        RandomHFlip(),
        RandomVFlip(),
        ColorJitter(),
        ToTensor(),
    ]
)

from bob.ip.binseg.data.utils import DelayedSample2TorchDataset
from bob.ip.binseg.data.stare import dataset as stare
dataset = DelayedSample2TorchDataset(stare.subsets("default")["train"],
        transform=_transforms)
