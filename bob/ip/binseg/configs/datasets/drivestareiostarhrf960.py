#!/usr/bin/env python
# coding=utf-8

"""COVD-CHASEDB1 (training set) for Vessel Segmentation
"""

from bob.ip.binseg.configs.datasets.drive960 import dataset as _drive
from bob.ip.binseg.configs.datasets.stare960 import dataset as _stare
from bob.ip.binseg.configs.datasets.hrf960 import dataset as _hrf
from bob.ip.binseg.configs.datasets.iostarvessel960 import dataset as _iostar

import torch.utils.data
dataset = torch.utils.data.ConcatDataset([_drive, _stare, _hrf, _iostar])
