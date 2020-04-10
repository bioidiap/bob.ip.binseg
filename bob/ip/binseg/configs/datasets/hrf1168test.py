#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""HRF (test set) for Vessel Segmentation

The database includes 15 images of each healthy, diabetic retinopathy (DR), and
glaucomatous eyes.  It contains 45 eye fundus images with a resolution of 3304
x 2336. One set of ground-truth vessel annotations is available.

* Reference: [HRF-2013]_
* Original resolution (height x width): 2336 x 3504
* Configuration resolution: 1168 x 1648 (after specific cropping and rescaling)
* Test samples: 30
* Split reference: [ORLANDO-2017]_
"""

from bob.ip.binseg.data.transforms import *
_transforms = Compose([Crop(0, 108, 2336, 3296), Resize((1168)), ToTensor()])

from bob.ip.binseg.data.utils import DelayedSample2TorchDataset
from bob.ip.binseg.data.hrf import dataset as hrf
dataset = DelayedSample2TorchDataset(hrf.subsets("default")["test"],
        transform=_transforms)
