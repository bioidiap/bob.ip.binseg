#!/usr/bin/env python
# coding=utf-8

"""RIM-ONE r3 (test set) for Optic Disc Segmentation

The dataset contains 159 stereo eye fundus images with a resolution of 2144 x
1424. The right part of the stereo image is disregarded. Two sets of
ground-truths for optic disc and optic cup are available. The first set is
commonly used for training and testing. The second set acts as a “human”
baseline.

* Reference: [RIMONER3-2015]_
* Original resolution (height x width): 1424 x 1072
* Configuration resolution: 1440 x 1088 (after padding)
* Test samples: 60
* Split reference: [MANINIS-2016]_
"""

from bob.ip.binseg.data.transforms import Pad

_transforms = [Pad((8, 8, 8, 8))]

from bob.ip.binseg.data.utils import SampleList2TorchDataset
from bob.ip.binseg.data.rimoner3 import dataset as rimoner3

dataset = SampleList2TorchDataset(
    rimoner3.subsets("optic-disc-exp1")["test"], transforms=_transforms
)
