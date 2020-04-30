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

from bob.ip.binseg.data.transforms import Crop, Pad, Resize
from bob.ip.binseg.configs.datasets import make_trainset as _maker

from bob.ip.binseg.data.drive import dataset as _raw_drive

_drive = _maker(
    _raw_drive.subsets("default")["train"],
    [Crop(75, 10, 416, 544), Pad((21, 0, 22, 0)), Resize(1168)],
    rotation_before=True,
)

from bob.ip.binseg.data.stare import dataset as _raw_stare

_stare = _maker(
    _raw_stare.subsets("ah")["train"],
    [Crop(50, 0, 500, 705), Resize(1168), Pad((1, 0, 1, 0))],
    rotation_before=True,
)

from bob.ip.binseg.data.chasedb1 import dataset as _raw_chase

_chase = _maker(
    _raw_chase.subsets("first-annotator")["train"],
    [Crop(140, 18, 680, 960), Resize(1168)],
    rotation_before=True,
)

from bob.ip.binseg.data.iostar import dataset as _raw_iostar

_iostar = _maker(
    _raw_iostar.subsets("vessel")["train"],
    [Crop(144, 0, 768, 1024), Pad((30, 0, 30, 0)), Resize(1168)],
    rotation_before=True,
)

from torch.utils.data import ConcatDataset
from bob.ip.binseg.configs.datasets.hrf.default import dataset as _baseline

# copy dictionary and replace only the augmented train dataset
dataset = dict(**_baseline)
dataset["__train__"] = ConcatDataset([_drive, _stare, _chase, _iostar])
