#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""HRF (training set) for Vessel Segmentation

The database includes 15 images of each healthy, diabetic retinopathy (DR), and
glaucomatous eyes.  It contains 45 eye fundus images with a resolution of 3304
x 2336. One set of ground-truth vessel annotations is available.

* Reference: [HRF-2013]_
* Original resolution (height x width): 2336 x 3504
* Configuration resolution: 608 x 608 (after specific padding and rescaling)
* Training samples: 15
* Split reference: [ORLANDO-2017]_
"""

from bob.ip.binseg.data.transforms import Pad, Resize
from bob.ip.binseg.configs.datasets.utils import DATA_AUGMENTATION as _DA
_transforms = [Pad((0, 345, 0, 345)), Resize(608)] + _DA

from bob.ip.binseg.data.utils import SampleList2TorchDataset
from bob.ip.binseg.data.hrf import dataset as hrf
dataset = SampleList2TorchDataset(hrf.subsets("default")["train"],
        transforms=_transforms)
