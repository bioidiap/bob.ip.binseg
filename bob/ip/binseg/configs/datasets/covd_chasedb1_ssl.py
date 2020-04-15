#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""COVD-CHASE-DB1 + SSL (training set) for Vessel Segmentation

* Configuration resolution (height x width): 960 x 960

The dataset available in this file is composed of DRIVE, STARE, IOSTAR
vessel and HRF (with annotated samples) and CHASE-DB1 without labels.
"""

from bob.ip.binseg.configs.datasets.covd_chasedb1 import dataset as _labelled
from bob.ip.binseg.configs.datasets.chasedb1 import dataset as _unlabelled
from bob.ip.binseg.data.utils import SSLDataset
dataset = SSLDataset(_labelled, _unlabelled)
