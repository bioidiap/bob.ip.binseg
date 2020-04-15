#!/usr/bin/env python
# coding=utf-8

"""COVD-HRF (training set) for Vessel Segmentation

* Configuration resolution: 1168 x 1648

The dataset available in this file is composed of DRIVE STARE, CHASE-DB1, and
IOSTAR vessel (with annotated samples).
"""

from bob.ip.binseg.data.transforms import *
from bob.ip.binseg.data.utils import SampleList2TorchDataset

from bob.ip.binseg.data.drive import dataset as _raw_drive
_drive_transforms = [
        RandomRotation(),
        Crop(75, 10, 416, 544),
        Pad((21, 0, 22, 0)),
        Resize(1168),
        RandomHFlip(),
        RandomVFlip(),
        ColorJitter(),
        ]
_drive = SampleList2TorchDataset(_raw_drive.subsets("default")["train"],
        transforms=_drive_transforms)

from bob.ip.binseg.data.stare import dataset as _raw_stare
_stare_transforms = [
        RandomRotation(),
        Crop(50, 0, 500, 705),
        Resize(1168),
        Pad((1, 0, 1, 0)),
        RandomHFlip(),
        RandomVFlip(),
        ColorJitter(),
        ]
_stare = SampleList2TorchDataset(_raw_stare.subsets("default")["train"],
        transforms=_stare_transforms)

from bob.ip.binseg.data.chasedb1 import dataset as _raw_chase
_chase_transforms = [
        RandomRotation(),
        Crop(140, 18, 680, 960),
        Resize(1168),
        RandomHFlip(),
        RandomVFlip(),
        ColorJitter(),
        ]
_chase = SampleList2TorchDataset(_raw_chase.subsets("default")["train"],
        transforms=_chase_transforms)

from bob.ip.binseg.data.iostar import dataset as _raw_iostar
_iostar_transforms = [
        RandomRotation(),
        Crop(144, 0, 768, 1024),
        Pad((30, 0, 30, 0)),
        Resize(1168),
        RandomHFlip(),
        RandomVFlip(),
        ColorJitter(),
        ]
_iostar = SampleList2TorchDataset(_raw_iostar.subsets("vessel")["train"],
        transforms=_iostar_transforms)

import torch.utils.data
dataset = torch.utils.data.ConcatDataset([_drive, _stare, _chase, _iostar])
