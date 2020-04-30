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

from bob.ip.binseg.data.transforms import CenterCrop, Pad, Resize
from bob.ip.binseg.configs.datasets import make_trainset as _maker

from bob.ip.binseg.data.drive import dataset as _raw_drive

_drive = _maker(
    _raw_drive.subsets("default")["train"],
    [CenterCrop((470, 544)), Pad((10, 9, 10, 8)), Resize(608)],
    rotation_before=True,
)

from bob.ip.binseg.data.chasedb1 import dataset as _raw_chase

_chase = _maker(
    _raw_chase.subsets("first-annotator")["train"],
    [CenterCrop((829, 960)), Resize(608)],
    rotation_before=True,
)

from bob.ip.binseg.data.iostar import dataset as _raw_iostar

_iostar = _maker(
    _raw_iostar.subsets("vessel")["train"],
    # n.b.: not the best fit, but what was there for Tim's work
    [Pad((81, 0, 81, 0)), Resize(608)],
)

from bob.ip.binseg.data.hrf import dataset as _raw_hrf

_hrf = _maker(
    _raw_hrf.subsets("default")["train"], [Pad((0, 345, 0, 345)), Resize(608)],
)

from torch.utils.data import ConcatDataset
from bob.ip.binseg.configs.datasets.stare.ah import dataset as _baseline

# copy dictionary and replace only the augmented train dataset
dataset = dict(**_baseline)
dataset["__train__"] = ConcatDataset([_drive, _chase, _iostar, _hrf])
