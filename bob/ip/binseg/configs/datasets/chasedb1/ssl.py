#!/usr/bin/env python
# coding=utf-8

"""COVD-CHASE-DB1 + SSL for Vessel Segmentation

* Configuration resolution: 960 x 960

The dataset available in this file is composed of DRIVE, STARE, IOSTAR vessel
and HRF (with annotated samples) and CHASE-DB1's "first-annotator" training set
without labels, for training, and CHASE-DB1's "first-annotator" test set, for
evaluation.

For details on datasets, consult:

* :py:mod:`bob.ip.binseg.data.stare`
* :py:mod:`bob.ip.binseg.data.drive`
* :py:mod:`bob.ip.binseg.data.chasedb1`
* :py:mod:`bob.ip.binseg.data.iostar`
* :py:mod:`bob.ip.binseg.data.hrf`
"""

from bob.ip.binseg.configs.datasets.chasedb1.covd import dataset as _labelled
from bob.ip.binseg.configs.datasets.chasedb1.first_annotator import (
    dataset as _baselines,
)
from bob.ip.binseg.data.utils import SSLDataset

dataset = {
    "train": SSLDataset(_labelled["train"], _baselines["train"]),
    "test": _baselines["test"],  # use always the same test set
}
