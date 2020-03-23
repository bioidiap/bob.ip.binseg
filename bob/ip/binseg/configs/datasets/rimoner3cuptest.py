#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""RIM-ONE r3 (test set) for Cup Segmentation

The dataset contains 159 stereo eye fundus images with a resolution of 2144 x
1424. The right part of the stereo image is disregarded. Two sets of
ground-truths for optic disc and optic cup are available. The first set is
commonly used for training and testing. The second set acts as a “human”
baseline.

* Reference: [RIMONER3-2015]_
* Original resolution (height x width): 1424 x 1072
* Configuration resolution: 1440 x 1088 (after padding)
* Test samples: 60
* Split reference: [MANINIS-2016]_
"""

from bob.db.rimoner3 import Database as RIMONER3
from bob.ip.binseg.data.transforms import *
from bob.ip.binseg.data.binsegdataset import BinSegDataset

#### Config ####

transforms = Compose([Pad((8, 8, 8, 8)), ToTensor()])

# bob.db.dataset init
bobdb = RIMONER3(protocol="default_cup")

# PyTorch dataset
dataset = BinSegDataset(bobdb, split="test", transform=transforms)
