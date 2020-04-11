#!/usr/bin/env python
# coding=utf-8

"""COVD-DRIVE (training set) for Vessel Segmentation
"""

from bob.ip.binseg.configs.datasets.stare544 import dataset as _stare
from bob.ip.binseg.configs.datasets.chasedb1544 import dataset as _chase
from bob.ip.binseg.configs.datasets.iostarvessel544 import dataset as _iostar
from bob.ip.binseg.configs.datasets.hrf544 import dataset as _hrf

import torch.utils.data
dataset = torch.utils.data.ConcatDataset([_stare, _chase, _hrf, _iostar])
