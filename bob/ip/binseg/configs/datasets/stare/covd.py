#!/usr/bin/env python
# coding=utf-8

"""COVD-STARE for Vessel Segmentation

* Configuration resolution: 704 x 608

The dataset available in this file is composed of DRIVE, CHASE-DB1, IOSTAR
vessel and HRF (with annotated samples).

For details on those datasets, consult:

* See :py:mod:`bob.ip.binseg.data.drive`
* See :py:mod:`bob.ip.binseg.data.chasedb1`
* See :py:mod:`bob.ip.binseg.data.iostar`
* See :py:mod:`bob.ip.binseg.data.hrf`
"""

from torch.utils.data import ConcatDataset
from bob.ip.binseg.configs.datasets import augment_subset as _augment
from bob.ip.binseg.configs.datasets.stare.mtest import dataset as _mtest
from bob.ip.binseg.configs.datasets.stare.ah import dataset as _baseline

dataset = dict(**_baseline)
dataset["__train__"] = ConcatDataset([
    _augment(_mtest["drive"], rotation_before=True),
    _augment(_mtest["chasedb1"], rotation_before=True),
    _augment(_mtest["hrf"], rotation_before=False),
    _augment(_mtest["iostar"], rotation_before=False),
    ])
dataset["train"] = ConcatDataset([
    _mtest["drive"],
    _mtest["chasedb1"],
    _mtest["hrf"],
    _mtest["iostar"],
    ])
