#!/usr/bin/env python
# coding=utf-8

"""COVD-HRF + SSL for Vessel Segmentation

* Configuration resolution: 1168 x 1648

The dataset available in this file is composed of DRIVE STARE, CHASE-DB1, and
IOSTAR vessel (with annotated samples), and HRF "default" training set, without
labels, for training, and HRF's "default" test set, for evaluation.

For details on datasets, consult:

* :py:mod:`bob.ip.binseg.data.stare`
* :py:mod:`bob.ip.binseg.data.drive`
* :py:mod:`bob.ip.binseg.data.chasedb1`
* :py:mod:`bob.ip.binseg.data.iostar`
* :py:mod:`bob.ip.binseg.data.hrf`
"""

from bob.ip.binseg.configs.datasets.hrf.covd import dataset as _covd
from bob.ip.binseg.configs.datasets.hrf.default import dataset as _baseline
from bob.ip.binseg.data.utils import SSLDataset

# copy dictionary and replace only the augmented train dataset
dataset = dict(**_covd)
dataset["__train__"] = SSLDataset(_covd["__train__"], _baseline["__train__"])
