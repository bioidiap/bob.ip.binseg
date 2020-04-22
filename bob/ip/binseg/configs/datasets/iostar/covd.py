#!/usr/bin/env python
# coding=utf-8

"""COVD-IOSTAR for Vessel Segmentation

* Configuration resolution: 1024 x 1024

The dataset available in this file is composed of DRIVE, STARE, CHASE-DB1, and
HRF (with annotated samples).

For details on those datasets, consult:

* See :py:mod:`bob.ip.binseg.data.drive`
* See :py:mod:`bob.ip.binseg.data.stare`
* See :py:mod:`bob.ip.binseg.data.chasedb1`
* See :py:mod:`bob.ip.binseg.data.hrf`
"""

from bob.ip.binseg.data.transforms import CenterCrop, Crop, Pad, Resize
from bob.ip.binseg.configs.datasets import make_trainset as _maker

from bob.ip.binseg.data.drive import dataset as _raw_drive

_drive = _maker(
    _raw_drive.subsets("default")["train"],
    [CenterCrop((540, 540)), Resize(1024)],
    rotation_before=True,
)

from bob.ip.binseg.data.stare import dataset as _raw_stare

_stare = _maker(
    _raw_stare.subsets("ah")["train"],
    [Pad((0, 32, 0, 32)), Resize(1024), CenterCrop(1024)],
    rotation_before=True,
)

from bob.ip.binseg.data.hrf import dataset as _raw_hrf

_hrf = _maker(
    _raw_hrf.subsets("default")["train"], [Pad((0, 584, 0, 584)), Resize(1024)],
)

from bob.ip.binseg.data.chasedb1 import dataset as _raw_chase

_chase = _maker(
    _raw_chase.subsets("first-annotator")["train"],
    [Crop(0, 18, 960, 960), Resize(1024)],
    rotation_before=True,
)

from torch.utils.data import ConcatDataset
from bob.ip.binseg.configs.datasets.iostar.vessel import dataset as _baseline

# copy dictionary and replace only the augmented train dataset
dataset = dict(**_baseline)
dataset["__train__"] = ConcatDataset([_drive, _stare, _hrf, _chase])
