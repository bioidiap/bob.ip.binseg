#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""COVD-STARE (training set) for Vessel Segmentation

* Configuration resolution: 704 x 608 (after padding)

The dataset available in this file is composed of DRIVE, CHASE-DB1, IOSTAR
vessel and HRF (with annotated samples).
"""

from bob.ip.binseg.data.transforms import *
from bob.ip.binseg.data.utils import SampleList2TorchDataset
from bob.ip.binseg.configs.datasets.utils import DATA_AUGMENTATION as _DA

from bob.ip.binseg.data.drive import dataset as _raw_drive
_drive_transforms = [
        RandomRotation(),
        CenterCrop((470, 544)),
        Pad((10, 9, 10, 8)),
        Resize(608),
        RandomHFlip(),
        RandomVFlip(),
        ColorJitter(),
        ]
_drive = SampleList2TorchDataset(_raw_drive.subsets("default")["train"],
        transforms=_drive_transforms)

from bob.ip.binseg.data.chasedb1 import dataset as _raw_chase
_chase_transforms = [
        RandomRotation(),
        CenterCrop((829, 960)),
        Resize(608),
        RandomHFlip(),
        RandomVFlip(),
        ColorJitter(),
        ]
_chase = SampleList2TorchDataset(_raw_chase.subsets("default")["train"],
        transforms=_chase_transforms)

from bob.ip.binseg.data.iostar import dataset as _raw_iostar
_iostar_transforms = [Pad((81, 0, 81, 0)), Resize(608)] + _DA
_iostar = SampleList2TorchDataset(_raw_iostar.subsets("vessel")["train"],
        transforms=_iostar_transforms)

from bob.ip.binseg.data.hrf import dataset as _raw_hrf
_hrf_transforms = [Pad((0, 345, 0, 345)), Resize(608)] + _DA
_hrf = SampleList2TorchDataset(_raw_hrf.subsets("default")["train"],
        transforms=_hrf_transforms)

import torch.utils.data
dataset = torch.utils.data.ConcatDataset([_drive, _chase, _iostar, _hrf])
