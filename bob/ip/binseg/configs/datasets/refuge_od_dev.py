#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""REFUGE (validation set) for Optic Disc Segmentation

The dataset consists of 1200 color fundus photographs, created for a MICCAI
challenge. The goal of the challenge is to evaluate and compare automated
algorithms for glaucoma detection and optic disc/cup segmentation on a common
dataset of retinal fundus images.

* Reference: [REFUGE-2018]_
* Original resolution (height x width): 1634 x 1634
* Configuration resolution: 1632 x 1632 (after center cropping)
* Validation samples: 400
* Split reference: [REFUGE-2018]_
"""

from bob.ip.binseg.data.transforms import CenterCrop
_transforms = [CenterCrop(1632)]

from bob.ip.binseg.data.utils import SampleList2TorchDataset
from bob.ip.binseg.data.refuge import dataset as refuge
dataset = SampleList2TorchDataset(refuge.subsets("optic-disc")["validation"],
        transforms=_transforms)
