#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""IOSTAR (test set) for Vessel Segmentation

The IOSTAR vessel segmentation dataset includes 30 images with a resolution of
1024 × 1024 pixels. All the vessels in this dataset are annotated by a group of
experts working in the field of retinal image analysis. Additionally the
dataset includes annotations for the optic disc and the artery/vein ratio.

* Reference: [IOSTAR-2016]_
* Original resolution (height x width): 1024 x 1024
* Configuration resolution: 1024 x 1024
* Training samples: 10
* Split reference: [MEYER-2017]_
"""

from bob.db.iostar import Database as IOSTAR
from bob.ip.binseg.data.transforms import *
from bob.ip.binseg.data.binsegdataset import BinSegDataset

#### Config ####

transforms = Compose([ToTensor()])

# bob.db.dataset init
bobdb = IOSTAR(protocol="default_vessel")

# PyTorch dataset
dataset = BinSegDataset(bobdb, split="test", transform=transforms)
