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

from bob.ip.binseg.configs.datasets.chasedb1.covd import dataset as _covd
from bob.ip.binseg.configs.datasets.chasedb1.first_annotator import (
    dataset as _baseline,
)
from bob.ip.binseg.data.utils import SSLDataset

# copy dictionary and replace only the augmented train dataset
dataset = dict(**_covd)
dataset["__train__"] = SSLDataset(_covd["__train__"], _baseline["__train__"])
