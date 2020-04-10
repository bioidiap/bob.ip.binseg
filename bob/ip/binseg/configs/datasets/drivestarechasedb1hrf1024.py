#!/usr/bin/env python
# coding=utf-8

"""COVD-IOSTAR (training set) for Vessel Segmentation
"""

from bob.ip.binseg.configs.datasets.drive1024 import dataset as _drive
from bob.ip.binseg.configs.datasets.stare1024 import dataset as _stare
from bob.ip.binseg.configs.datasets.hrf1024 import dataset as _hrf
from bob.ip.binseg.configs.datasets.chasedb11024 import dataset as _chase

import torch.utils.data
dataset = torch.utils.data.ConcatDataset([_drive, _stare, _hrf, _chase])
