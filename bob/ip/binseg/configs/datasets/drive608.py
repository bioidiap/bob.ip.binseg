#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""DRIVE (training set) for Vessel Segmentation

The DRIVE database has been established to enable comparative studies on
segmentation of blood vessels in retinal images.

* Reference: [DRIVE-2004]_
* Original resolution (height x width): 584 x 565
* This configuration resolution: 608 x 608 (center-crop)
* Training samples: 20
* Split reference: [DRIVE-2004]_
"""

from bob.ip.binseg.data.transforms import *
_transforms = Compose(
    [
        RandomRotation(),
        CenterCrop((470, 544)),
        Pad((10, 9, 10, 8)),
        Resize(608),
        RandomHFlip(),
        RandomVFlip(),
        ColorJitter(),
        ToTensor(),
    ]
)

from bob.ip.binseg.data.utils import DelayedSample2TorchDataset
from bob.ip.binseg.data.drive import dataset as drive
dataset = DelayedSample2TorchDataset(drive.subsets("default")["train"],
        transform=_transforms)
