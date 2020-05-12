#!/usr/bin/env python
# coding=utf-8

"""COVD-CHASEDB1 for Vessel Segmentation

* Configuration resolution (height x width): 960 x 960

The dataset available in this file is composed of DRIVE, STARE, IOSTAR
vessel and HRF (with annotated samples).

For details on those datasets, consult:

* See :py:mod:`bob.ip.binseg.data.drive`
* See :py:mod:`bob.ip.binseg.data.stare`
* See :py:mod:`bob.ip.binseg.data.iostar`
* See :py:mod:`bob.ip.binseg.data.hrf`
"""

from torch.utils.data import ConcatDataset
from bob.ip.binseg.configs.datasets import augment_subset as _augment
from bob.ip.binseg.configs.datasets.chasedb1.mtest import dataset as _mtest
from bob.ip.binseg.configs.datasets.chasedb1.first_annotator import (
    dataset as _baseline,
    second_annotator,
)

dataset = dict(**_baseline)
dataset["__train__"] = ConcatDataset(
    [
        _augment(_mtest["drive"], rotation_before=True),
        _augment(_mtest["stare"], rotation_before=True),
        _augment(_mtest["hrf"], rotation_before=False),
        _augment(_mtest["iostar"], rotation_before=False),
    ]
)
dataset["train"] = ConcatDataset(
    [_mtest["drive"], _mtest["stare"], _mtest["hrf"], _mtest["iostar"],]
)
dataset["__valid__"] = dataset["train"]
