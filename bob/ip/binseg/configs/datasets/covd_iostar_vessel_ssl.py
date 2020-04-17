#!/usr/bin/env python
# coding=utf-8

"""COVD-IOSTAR + SSL (training set) for Vessel Segmentation

* Configuration resolution: 1024 x 1024

The dataset available in this file is composed of DRIVE, STARE, CHASE-DB1, and
HRF (with annotated samples) and IOSTAR without labels.
"""

from bob.ip.binseg.configs.datasets.covd_iostar_vessel import dataset as _labelled
from bob.ip.binseg.configs.datasets.iostar_vessel import dataset as _unlabelled
from bob.ip.binseg.data.utils import SSLDataset

dataset = SSLDataset(_labelled, _unlabelled)
