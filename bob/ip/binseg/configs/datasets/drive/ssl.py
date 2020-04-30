#!/usr/bin/env python
# coding=utf-8

"""COVD-DRIVE + SSL for Vessel Segmentation

* Configuration resolution: 544 x 544

The dataset available in this file is composed of STARE, CHASE-DB1, IOSTAR
vessel and HRF (with annotated samples) and DRIVE "default" training set
without labels, for training, and DRIVE's "default" test set, for evaluation.

For details on datasets, consult:

* :py:mod:`bob.ip.binseg.data.stare`
* :py:mod:`bob.ip.binseg.data.drive`
* :py:mod:`bob.ip.binseg.data.chasedb1`
* :py:mod:`bob.ip.binseg.data.iostar`
* :py:mod:`bob.ip.binseg.data.hrf`
"""

from bob.ip.binseg.configs.datasets.drive.covd import dataset as _covd
from bob.ip.binseg.configs.datasets.drive.default import dataset as _baseline
from bob.ip.binseg.data.utils import SSLDataset

# copy dictionary and replace only the augmented train dataset
dataset = dict(**_covd)
dataset["__train__"] = SSLDataset(_covd["__train__"], _baseline["__train__"])
