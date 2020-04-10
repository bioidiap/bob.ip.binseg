#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""HRF (SSL training set) for Vessel Segmentation

The database includes 15 images of each healthy, diabetic retinopathy (DR), and
glaucomatous eyes.  It contains 45 eye fundus images with a resolution of 3504
x 2336. One set of ground-truth vessel annotations is available.

* Reference: [HRF-2013]_
* Configuration resolution: 1168 x 1648

The dataset available in this file is composed of STARE, CHASE-DB1, IOSTAR
vessel and CHASE-DB1 (with annotated samples) and HRF without labels.
"""

from bob.ip.binseg.configs.datasets.drivestarechasedb1iostar1168 import dataset as _labelled
from bob.ip.binseg.configs.datasets.hrf1168 import dataset as _unlabelled
from bob.ip.binseg.data.utils import SSLDataset
dataset = SSLDataset(_labelled, _unlabelled)
