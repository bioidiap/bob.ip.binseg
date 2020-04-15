#!/usr/bin/env python
# coding=utf-8

"""COVD-IOSTAR (training set) for Vessel Segmentation

* Configuration resolution: 1024 x 1024

The dataset available in this file is composed of DRIVE, STARE, CHASE-DB1, and
HRF (with annotated samples).
"""

from bob.ip.binseg.data.transforms import *
from bob.ip.binseg.data.utils import SampleList2TorchDataset
from bob.ip.binseg.configs.datasets.utils import DATA_AUGMENTATION as _DA

from bob.ip.binseg.data.drive import dataset as _raw_drive
_drive_transforms = [
        RandomRotation(),
        CenterCrop((540, 540)),
        Resize(1024),
        RandomHFlip(),
        RandomVFlip(),
        ColorJitter(),
        ]
_drive = SampleList2TorchDataset(_raw_drive.subsets("default")["train"],
        transforms=_drive_transforms)


from bob.ip.binseg.data.stare import dataset as _raw_stare
_stare_transforms = [
        RandomRotation(),
        Pad((0, 32, 0, 32)),
        Resize(1024),
        CenterCrop(1024),
        RandomHFlip(),
        RandomVFlip(),
        ColorJitter(),
    ]
_stare = SampleList2TorchDataset(_raw_stare.subsets("default")["train"],
        transforms=_stare_transforms)


from bob.ip.binseg.data.hrf import dataset as _raw_hrf
_hrf_transforms = [Pad((0, 584, 0, 584)), Resize(1024)] + _DA
_hrf = SampleList2TorchDataset(_raw_hrf.subsets("default")["train"],
        transforms=_hrf_transforms)

from bob.ip.binseg.data.chasedb1 import dataset as _chase_raw
_chase_transforms = [
        RandomRotation(),
        Crop(0, 18, 960, 960),
        Resize(1024),
        RandomHFlip(),
        RandomVFlip(),
        ColorJitter(),
        ]
_chase = SampleList2TorchDataset(_chase_raw.subsets("default")["train"],
        transforms=_chase_transforms)

import torch.utils.data
dataset = torch.utils.data.ConcatDataset([_drive, _stare, _hrf, _chase])
