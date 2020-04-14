#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""DRISHTI-GS1 (training set) for Cup Segmentation

Drishti-GS is a dataset meant for validation of segmenting OD, cup and
detecting notching.  The images in the Drishti-GS dataset have been collected
and annotated by Aravind Eye hospital, Madurai, India. This dataset is of a
single population as all subjects whose eye images are part of this dataset are
Indians.

The dataset is divided into two: a training set and a testing set of images.
Training images (50) are provided with groundtruths for OD and Cup segmentation
and notching information.

* Reference (includes split): [DRISHTIGS1-2014]_
* Original resolution (height x width): varying (min: 1749 x 2045, max: 1845 x
  2468)
* Configuration resolution: 1760 x 2048 (after center cropping)
* Training samples: 50
"""

from bob.ip.binseg.data.transforms import CenterCrop
from bob.ip.binseg.configs.datasets.utils import DATA_AUGMENTATION as _DA
_transforms = [CenterCrop((1760, 2048))] + _DA

from bob.ip.binseg.data.utils import SampleList2TorchDataset
from bob.ip.binseg.data.drishtigs1 import dataset as drishtigs1
dataset = SampleList2TorchDataset(drishtigs1.subsets("optic-cup-all")["train"],
        transforms=_transforms)
