#!/usr/bin/env python
# coding=utf-8

"""IOSTAR (training set) for Optic Disc Segmentation

The IOSTAR vessel segmentation dataset includes 30 images with a resolution of
1024 × 1024 pixels. All the vessels in this dataset are annotated by a group of
experts working in the field of retinal image analysis. Additionally the
dataset includes annotations for the optic disc and the artery/vein ratio.

* Reference: [IOSTAR-2016]_
* Original resolution (height x width): 1024 x 1024
* Configuration resolution: 1024 x 1024
* Test samples: 10
* Split reference: [MEYER-2017]_
"""

from bob.ip.binseg.data.utils import SampleList2TorchDataset
from bob.ip.binseg.data.iostar import dataset as iostar
dataset = SampleList2TorchDataset(iostar.subsets("optic-disc")["test"])
