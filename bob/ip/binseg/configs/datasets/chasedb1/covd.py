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

from bob.ip.binseg.data.transforms import CenterCrop, Pad, Resize
from bob.ip.binseg.configs.datasets import make_trainset as _maker

from bob.ip.binseg.data.drive import dataset as _raw_drive

_drive = _maker(
    _raw_drive.subsets("default")["train"],
    [CenterCrop((544, 544)), Resize(960)],
    rotation_before=True,
)

from bob.ip.binseg.data.stare import dataset as _raw_stare

_stare = _maker(
    _raw_stare.subsets("ah")["train"],
    [Pad((0, 32, 0, 32)), Resize(960), CenterCrop(960)],
    rotation_before=True,
)

from bob.ip.binseg.data.hrf import dataset as _raw_hrf

_hrf = _maker(
    _raw_hrf.subsets("default")["train"], [Pad((0, 584, 0, 584)), Resize(960)],
)

from bob.ip.binseg.data.iostar import dataset as _raw_iostar

_iostar = _maker(_raw_iostar.subsets("vessel")["train"], [Resize(960)])

from torch.utils.data import ConcatDataset
from bob.ip.binseg.configs.datasets.chasedb1.first_annotator import (
    dataset as _baselines,
)

dataset = {
    "train": ConcatDataset([_drive, _stare, _hrf, _iostar]),
    "test": _baselines["test"],  # use the same test set always
}
