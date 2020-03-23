#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""HRF (training set) for Vessel Segmentation

The database includes 15 images of each healthy, diabetic retinopathy (DR), and
glaucomatous eyes.  It contains 45 eye fundus images with a resolution of 3304
x 2336. One set of ground-truth vessel annotations is available.

* Reference: [HRF-2013]_
* Original resolution (height x width): 2336 x 3504
* Configuration resolution: 1168 x 1648 (after specific cropping and rescaling)
* Training samples: 15
* Split reference: [ORLANDO-2017]_
"""

from bob.db.hrf import Database as HRF
from bob.ip.binseg.data.transforms import *
from bob.ip.binseg.data.binsegdataset import BinSegDataset

#### Config ####

transforms = Compose(
    [
        Crop(0, 108, 2336, 3296),  #(upper, left, height, width)
        Resize((1168)),  # applies to the smaller edge
        RandomHFlip(),
        RandomVFlip(),
        RandomRotation(),
        ColorJitter(),
        ToTensor(),
    ]
)

# bob.db.dataset init
bobdb = HRF(protocol="default")

# PyTorch dataset
dataset = BinSegDataset(bobdb, split="train", transform=transforms)
