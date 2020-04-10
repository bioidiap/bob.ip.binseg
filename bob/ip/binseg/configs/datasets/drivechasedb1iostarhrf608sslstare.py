#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""STARE (SSL training set) for Vessel Segmentation

A subset of the original STARE dataset contains 20 annotated eye fundus images
with a resolution of 605 x 700 (height x width). Two sets of ground-truth
vessel annotations are available. The first set by Adam Hoover is commonly used
for training and testing. The second set by Valentina Kouznetsova acts as a
“human” baseline.

* Reference: [STARE-2000]_
* Configuration resolution: 704 x 608 (after padding)

The dataset available in this file is composed of DRIVE, CHASE-DB1, IOSTAR
vessel and HRF (with annotated samples) and STARE without labels.
"""

# Labelled bits
import torch.utils.data

from bob.ip.binseg.configs.datasets.drive608 import dataset as _drive
from bob.ip.binseg.configs.datasets.chasedb1608 import dataset as _chase
from bob.ip.binseg.configs.datasets.iostarvessel608 import dataset as _iostar
from bob.ip.binseg.configs.datasets.hrf608 import dataset as _hrf
_labelled = torch.utils.data.ConcatDataset([_drive, _chase, _iostar, _hrf])

# Use STARE without labels in this setup
from bob.ip.binseg.configs.datasets.stare import dataset as _unlabelled

from bob.ip.binseg.data.utils import SSLDataset
dataset = SSLDataset(_labelled, _unlabelled)
