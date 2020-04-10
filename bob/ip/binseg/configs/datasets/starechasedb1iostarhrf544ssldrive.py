#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""COVD-DRIVE + SSL (training set) for Vessel Segmentation

* Configuration resolution: 544 x 544

The dataset available in this file is composed of STARE, CHASE-DB1, IOSTAR
vessel and HRF (with annotated samples) and DRIVE without labels.
"""

from bob.ip.binseg.configs.datasets.starechasedb1iostarhrf544 import dataset as _unlabelled
from bob.ip.binseg.configs.datasets.drive import dataset as _unlabelled
from bob.ip.binseg.data.utils import SSLDataset
dataset = SSLDataset(_labelled, _unlabelled)
