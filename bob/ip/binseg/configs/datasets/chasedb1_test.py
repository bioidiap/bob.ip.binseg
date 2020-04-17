#!/usr/bin/env python
# coding=utf-8

"""CHASE-DB1 (test set) for Vessel Segmentation

The CHASE_DB1 is a retinal vessel reference dataset acquired from multiethnic
school children. This database is a part of the Child Heart and Health Study in
England (CHASE), a cardiovascular health survey in 200 primary schools in
London, Birmingham, and Leicester. The ocular imaging was carried out in
46 schools and demonstrated associations between retinal vessel tortuosity and
early risk factors for cardiovascular disease in over 1000 British primary
school children of different ethnic origin. The retinal images of both of the
eyes of each child were recorded with a hand-held Nidek NM-200-D fundus camera.
The images were captured at 30 degrees FOV camera. The dataset of images are
characterized by having nonuniform back-ground illumination, poor contrast of
blood vessels as compared with the background and wider arteriolars that have a
bright strip running down the centre known as the central vessel reflex.

* Reference: [CHASEDB1-2012]_
* Original resolution (height x width): 960 x 999
* Configuration resolution: 960 x 960 (after hand-specified crop)
* Test samples: 20
* Split reference: [CHASEDB1-2012]_
"""

from bob.ip.binseg.data.transforms import Crop

_transforms = [Crop(0, 18, 960, 960)]  # (upper, left, height, width)

from bob.ip.binseg.data.utils import SampleList2TorchDataset
from bob.ip.binseg.data.chasedb1 import dataset as chasedb1

dataset = SampleList2TorchDataset(
    chasedb1.subsets("default")["test"], transforms=_transforms
)
