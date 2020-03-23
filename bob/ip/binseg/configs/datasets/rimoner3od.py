#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""RIM-ONE r3 (training set) for Optic Disc Segmentation

The dataset contains 159 stereo eye fundus images with a resolution of 2144 x
1424. The right part of the stereo image is disregarded. Two sets of
ground-truths for optic disc and optic cup are available. The first set is
commonly used for training and testing. The second set acts as a “human”
baseline.

* Reference: [RIMONER3-2015]_
* Original resolution (height x width): 1424 x 1072
* Configuration resolution: 1440 x 1088 (after padding)
* Training samples: 99
* Split reference: [MANINIS-2016]_
"""

from bob.db.rimoner3 import Database as RIMONER3
from bob.ip.binseg.data.transforms import *
from bob.ip.binseg.data.binsegdataset import BinSegDataset

#### Config ####

transforms = Compose(
    [
        Pad((8, 8, 8, 8)),
        RandomHFlip(),
        RandomVFlip(),
        RandomRotation(),
        ColorJitter(),
        ToTensor(),
    ]
)

# bob.db.dataset init
bobdb = RIMONER3(protocol="default_od")

# PyTorch dataset
dataset = BinSegDataset(bobdb, split="train", transform=transforms)
