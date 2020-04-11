#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""COVD-STARE (training set) for Vessel Segmentation

* Configuration resolution: 704 x 608 (after padding)

The dataset available in this file is composed of DRIVE, CHASE-DB1, IOSTAR
vessel and HRF (with annotated samples).
"""

from bob.ip.binseg.configs.datasets.drive608 import dataset as _drive
from bob.ip.binseg.configs.datasets.chasedb1608 import dataset as _chase
from bob.ip.binseg.configs.datasets.iostarvessel608 import dataset as _iostar
from bob.ip.binseg.configs.datasets.hrf608 import dataset as _hrf

import torch.utils.data
dataset = torch.utils.data.ConcatDataset([_drive, _chase, _iostar, _hrf])
