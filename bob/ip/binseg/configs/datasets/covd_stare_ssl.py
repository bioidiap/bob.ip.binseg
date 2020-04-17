#!/usr/bin/env python
# coding=utf-8

"""COVD-STARE + SSL (training set) for Vessel Segmentation

* Configuration resolution: 704 x 608

The dataset available in this file is composed of DRIVE, CHASE-DB1, IOSTAR
vessel and HRF (with annotated samples) and STARE without labels.
"""

from bob.ip.binseg.configs.datasets.covd_stare import dataset as _labelled
from bob.ip.binseg.configs.datasets.stare import dataset as _unlabelled
from bob.ip.binseg.data.utils import SSLDataset

dataset = SSLDataset(_labelled, _unlabelled)
