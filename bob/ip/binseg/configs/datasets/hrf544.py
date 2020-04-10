#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""HRF (training set) for Vessel Segmentation

The database includes 15 images of each healthy, diabetic retinopathy (DR), and
glaucomatous eyes.  It contains 45 eye fundus images with a resolution of 3304
x 2336. One set of ground-truth vessel annotations is available.

* Reference: [HRF-2013]_
* Original resolution (height x width): 2336 x 3504
* Configuration resolution: 544 x 544 (after specific padding and rescaling)
* Test samples: 30
* Split reference: [ORLANDO-2017]_
"""

from bob.ip.binseg.data.transforms import *
_transforms = Compose(
    [
        Resize((363)),
        Pad((0, 90, 0, 91)),
        RandomRotation(),
        RandomHFlip(),
        RandomVFlip(),
        ColorJitter(),
        ToTensor(),
    ]
)

from bob.ip.binseg.data.utils import DelayedSample2TorchDataset
from bob.ip.binseg.data.hrf import dataset as hrf
dataset = DelayedSample2TorchDataset(hrf.subsets("default")["train"],
        transform=_transforms)
