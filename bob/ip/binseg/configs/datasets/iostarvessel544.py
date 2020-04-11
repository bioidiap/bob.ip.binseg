#!/usr/bin/env python
# coding=utf-8

"""IOSTAR (training set) for Vessel Segmentation

The IOSTAR vessel segmentation dataset includes 30 images with a resolution of
1024 × 1024 pixels. All the vessels in this dataset are annotated by a group of
experts working in the field of retinal image analysis. Additionally the
dataset includes annotations for the optic disc and the artery/vein ratio.

* Reference: [IOSTAR-2016]_
* Original resolution (height x width): 1024 x 1024
* Configuration resolution: 544 x 544
* Training samples: 20
* Split reference: [MEYER-2017]_
"""

from bob.ip.binseg.data.transforms import Resize
from bob.ip.binseg.configs.datasets.utils import DATA_AUGMENTATION as _DA
_transforms = [Resize(544)] + _DA

from bob.ip.binseg.data.utils import SampleList2TorchDataset
from bob.ip.binseg.data.iostar import dataset as iostar
dataset = SampleList2TorchDataset(iostar.subsets("vessel")["train"],
        transforms=_transforms)
