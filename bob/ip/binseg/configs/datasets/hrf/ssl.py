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

from bob.ip.binseg.configs.datasets.hrf.covd import dataset as _labelled
from bob.ip.binseg.configs.datasets.hrf.default import dataset as _baselines
from bob.ip.binseg.data.utils import SSLDataset

dataset = {
    "train": SSLDataset(_labelled["train"], _baselines["train"]),
    "test": _baselines["test"],  # use always the same test set
}
