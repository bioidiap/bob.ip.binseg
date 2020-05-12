#!/usr/bin/env python
# coding=utf-8

"""COVD-HRF for Vessel Segmentation

* Configuration resolution: 1168 x 1648

The dataset available in this file is composed of DRIVE STARE, CHASE-DB1, and
IOSTAR vessel (with annotated samples).

For details on those datasets, consult:

* See :py:mod:`bob.ip.binseg.data.drive`
* See :py:mod:`bob.ip.binseg.data.stare`
* See :py:mod:`bob.ip.binseg.data.chasedb1`
* See :py:mod:`bob.ip.binseg.data.iostar`
"""

from torch.utils.data import ConcatDataset
from bob.ip.binseg.configs.datasets import augment_subset as _augment
from bob.ip.binseg.configs.datasets.hrf.mtest import dataset as _mtest
from bob.ip.binseg.configs.datasets.hrf.default import dataset as _baseline

dataset = dict(**_baseline)
dataset["__train__"] = ConcatDataset(
    [
        _augment(_mtest["drive (train)"], rotation_before=True),
        _augment(_mtest["drive (test)"], rotation_before=True),
        _augment(_mtest["stare (train)"], rotation_before=True),
        _augment(_mtest["stare (test)"], rotation_before=True),
        _augment(_mtest["chasedb1 (train)"], rotation_before=True),
        _augment(_mtest["chasedb1 (test)"], rotation_before=True),
        _augment(_mtest["iostar (train)"], rotation_before=True),
        _augment(_mtest["iostar (test)"], rotation_before=True),
    ]
)
dataset["train"] = ConcatDataset(
    [
        _mtest["drive (train)"],
        _mtest["drive (test)"],
        _mtest["stare (train)"],
        _mtest["stare (test)"],
        _mtest["chasedb1 (train)"],
        _mtest["chasedb1 (test)"],
        _mtest["iostar (train)"],
        _mtest["iostar (test)"],
    ]
)
dataset["__valid__"] = dataset["train"]
