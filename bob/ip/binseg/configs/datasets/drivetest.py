#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""DRIVE (test set) for Vessel Segmentation

The DRIVE database has been established to enable comparative studies on
segmentation of blood vessels in retinal images.

* Reference: [DRIVE-2004]_
* Original resolution (height x width): 584 x 565
* Configuration resolution: 544 x 544 (after center-crop)
* Test samples: 20
* Split reference: [DRIVE-2004]_
"""

from bob.ip.binseg.data.transforms import *
_transforms = Compose([CenterCrop((544, 544)), ToTensor()])

from bob.ip.binseg.data.utils import DelayedSample2TorchDataset
from bob.ip.binseg.data.drive import dataset as drive
dataset = DelayedSample2TorchDataset(drive.subsets("default")["test"],
        transform=_transforms)
