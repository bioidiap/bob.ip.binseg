#!/usr/bin/env python
# coding=utf-8

"""IOSTAR (training set) for Vessel Segmentation

The IOSTAR vessel segmentation dataset includes 30 images with a resolution of
1024 × 1024 pixels. All the vessels in this dataset are annotated by a group of
experts working in the field of retinal image analysis. Additionally the
dataset includes annotations for the optic disc and the artery/vein ratio.

* Reference: [IOSTAR-2016]_
* Original resolution (height x width): 1024 x 1024
* Configuration resolution: 960 x 960
* Training samples: 20
* Split reference: [MEYER-2017]_
"""

from bob.ip.binseg.data.transforms import *
_transforms = Compose(
    [
        Resize(960),
        RandomHFlip(),
        RandomVFlip(),
        RandomRotation(),
        ColorJitter(),
        ToTensor(),
    ]
)

from bob.ip.binseg.data.utils import DelayedSample2TorchDataset
from bob.ip.binseg.data.iostar import dataset as iostar
dataset = DelayedSample2TorchDataset(iostar.subsets("vessel")["train"],
        transform=_transforms)
