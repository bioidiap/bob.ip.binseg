#!/usr/bin/env python
# coding=utf-8

"""COVD-HRF (training set) for Vessel Segmentation
"""

from bob.ip.binseg.configs.datasets.drive1168 import dataset as _drive
from bob.ip.binseg.configs.datasets.stare1168 import dataset as _stare
from bob.ip.binseg.configs.datasets.chasedb11168 import dataset as _chase
from bob.ip.binseg.configs.datasets.iostarvessel1168 import dataset as _iostar

import torch.utils.data
dataset = torch.utils.data.ConcatDataset([_drive, _stare, _chase, _iostar])
