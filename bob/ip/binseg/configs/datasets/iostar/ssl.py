#!/usr/bin/env python
# coding=utf-8

"""COVD-IOSTAR + SSL for Vessel Segmentation

* Configuration resolution: 1024 x 1024

The dataset available in this file is composed of DRIVE STARE, HRF, and
CHASE-DB1 (with annotated samples), and IOSTAR "vessel" training set, without
labels, for training, and IOSTAR's "vessel" test set, for evaluation.

For details on datasets, consult:

* :py:mod:`bob.ip.binseg.data.stare`
* :py:mod:`bob.ip.binseg.data.drive`
* :py:mod:`bob.ip.binseg.data.hrf`
* :py:mod:`bob.ip.binseg.data.chasedb1`
* :py:mod:`bob.ip.binseg.data.iostar`
"""

from bob.ip.binseg.configs.datasets.iostar.covd import dataset as _covd
from bob.ip.binseg.configs.datasets.iostar.vessel import dataset as _baseline
from bob.ip.binseg.data.utils import SSLDataset

# copy dictionary and replace only the augmented train dataset
dataset = dict(**_covd)
dataset["__train__"] = SSLDataset(_covd["__train__"], _baseline["__train__"])
