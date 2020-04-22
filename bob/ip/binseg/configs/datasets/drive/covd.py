#!/usr/bin/env python
# coding=utf-8

"""COVD-DRIVE for Vessel Segmentation

* Configuration resolution: 544 x 544

The dataset available in this file is composed of STARE, CHASE-DB1, IOSTAR
vessel and HRF (with annotated samples).

For details on those datasets, consult:

* See :py:mod:`bob.ip.binseg.data.stare`
* See :py:mod:`bob.ip.binseg.data.chasedb1`
* See :py:mod:`bob.ip.binseg.data.iostar`
* See :py:mod:`bob.ip.binseg.data.hrf`
"""

from bob.ip.binseg.data.transforms import Resize, Pad, Crop
from bob.ip.binseg.configs.datasets import make_trainset as _maker

from bob.ip.binseg.data.stare import dataset as _raw_stare

_stare = _maker(
    _raw_stare.subsets("ah")["train"],
    [Resize(471), Pad((0, 37, 0, 36))],
    rotation_before=True,
)

from bob.ip.binseg.data.chasedb1 import dataset as _raw_chase

_chase = _maker(
    _raw_chase.subsets("first-annotator")["train"],
    [Resize(544), Crop(0, 12, 544, 544)],
)

from bob.ip.binseg.data.iostar import dataset as _raw_iostar

_iostar = _maker(_raw_iostar.subsets("vessel")["train"], [Resize(544)],)

from bob.ip.binseg.data.hrf import dataset as _raw_hrf

_hrf = _maker(
    _raw_hrf.subsets("default")["train"], [Resize((363)), Pad((0, 90, 0, 91))],
)

from torch.utils.data import ConcatDataset
from bob.ip.binseg.configs.datasets.drive.default import dataset as _baseline

# copy dictionary and replace only the augmented train dataset
dataset = dict(**_baseline)
dataset["__train__"] = ConcatDataset([_stare, _chase, _iostar, _hrf])
