#!/usr/bin/env python
# coding=utf-8

"""COVD-DRIVE (training set) for Vessel Segmentation

* Configuration resolution: 544 x 544

The dataset available in this file is composed of STARE, CHASE-DB1, IOSTAR
vessel and HRF (with annotated samples).
"""

from bob.ip.binseg.data.transforms import *
from bob.ip.binseg.data.utils import SampleList2TorchDataset
from bob.ip.binseg.configs.datasets.augmentation import (
    DEFAULT as _DA,
    DEFAULT_WITHOUT_ROTATION as _DA_NOROT,
    ROTATION as _ROT,
)

from bob.ip.binseg.data.stare import dataset as _raw_stare

_stare_transforms = _ROT + [Resize(471), Pad((0, 37, 0, 36))] + _DA_NOROT
_stare = SampleList2TorchDataset(
    _raw_stare.subsets("default")["train"], transforms=_stare_transforms
)

from bob.ip.binseg.data.chasedb1 import dataset as _raw_chase

_chase_transforms = [Resize(544), Crop(0, 12, 544, 544)] + _DA
_chase = SampleList2TorchDataset(
    _raw_chase.subsets("default")["train"], transforms=_chase_transforms
)

from bob.ip.binseg.data.iostar import dataset as _raw_iostar

_iostar_transforms = [Resize(544)] + _DA
_iostar = SampleList2TorchDataset(
    _raw_iostar.subsets("vessel")["train"], transforms=_iostar_transforms
)

from bob.ip.binseg.data.hrf import dataset as _raw_hrf

_hrf_transforms = [Resize((363)), Pad((0, 90, 0, 91))] + _DA
_hrf = SampleList2TorchDataset(
    _raw_hrf.subsets("default")["train"], transforms=_hrf_transforms
)

import torch.utils.data

dataset = torch.utils.data.ConcatDataset([_stare, _chase, _iostar, _hrf])
