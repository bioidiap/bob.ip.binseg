#!/usr/bin/env python
# coding=utf-8

"""COVD-CHASEDB1 (training set) for Vessel Segmentation

* Configuration resolution (height x width): 960 x 960

The dataset available in this file is composed of STARE, CHASE-DB1, IOSTAR
vessel and HRF (with annotated samples).
"""

from bob.ip.binseg.data.transforms import *
from bob.ip.binseg.configs.datasets.utils import DATA_AUGMENTATION as _DA
from bob.ip.binseg.data.utils import SampleList2TorchDataset

from bob.ip.binseg.data.drive import dataset as _raw_drive
_drive_transforms = [
        RandomRotation(),
        CenterCrop((544, 544)),
        Resize(960),
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
        Resize(960),
        CenterCrop(960),
        RandomHFlip(),
        RandomVFlip(),
        ColorJitter(),
    ]
_stare = SampleList2TorchDataset(_raw_stare.subsets("default")["train"],
        transforms=_stare_transforms)

from bob.ip.binseg.data.hrf import dataset as _raw_hrf
_hrf_transforms = [Pad((0, 584, 0, 584)), Resize(960)] + _DA
_hrf = SampleList2TorchDataset(_raw_hrf.subsets("default")["train"],
        transforms=_hrf_transforms)

from bob.ip.binseg.data.iostar import dataset as _raw_iostar
_iostar_transforms = [Resize(960)] + _DA
_iostar = SampleList2TorchDataset(_raw_iostar.subsets("vessel")["train"],
        transforms=_iostar_transforms)

import torch.utils.data
dataset = torch.utils.data.ConcatDataset([_drive, _stare, _hrf, _iostar])
