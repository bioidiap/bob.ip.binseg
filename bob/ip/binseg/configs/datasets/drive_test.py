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

from bob.ip.binseg.data.transforms import CenterCrop
_transforms = [CenterCrop((544, 544))]

from bob.ip.binseg.data.utils import SampleList2TorchDataset
from bob.ip.binseg.data.drive import dataset as drive
dataset = SampleList2TorchDataset(drive.subsets("default")["test"],
        transforms=_transforms)
