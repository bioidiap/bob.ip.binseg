#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""HRF (training set) for Vessel Segmentation

The database includes 15 images of each healthy, diabetic retinopathy (DR), and
glaucomatous eyes.  It contains 45 eye fundus images with a resolution of 3504
x 2336. One set of ground-truth vessel annotations is available.

* Reference: [HRF-2013]_
* Original resolution (height x width): 2336 x 3504
* Configuration resolution: 2336 x 3296 (after specific cropping and rescaling)
* Training samples: 15
* Split reference: [ORLANDO-2017]_
"""

from bob.ip.binseg.data.transforms import *
_transforms = Compose(
    [
        Crop(0, 108, 2336, 3296),
        RandomHFlip(),
        RandomVFlip(),
        RandomRotation(),
        ColorJitter(),
        ToTensor(),
    ]
)

from bob.ip.binseg.data.utils import DelayedSample2TorchDataset
from bob.ip.binseg.data.hrf import dataset as hrf
dataset = DelayedSample2TorchDataset(hrf.subsets("default")["train"],
        transform=_transforms)
