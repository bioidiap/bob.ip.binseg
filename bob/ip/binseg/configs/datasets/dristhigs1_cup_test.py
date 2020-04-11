#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""DRISHTI-GS1 (test set) for Cup Segmentation

Drishti-GS is a dataset meant for validation of segmenting OD, cup and
detecting notching.  The images in the Drishti-GS dataset have been collected
and annotated by Aravind Eye hospital, Madurai, India. This dataset is of a
single population as all subjects whose eye images are part of this dataset are
Indians.

The dataset is divided into two: a training set and a testing set of images.
Training images (50) are provided with groundtruths for OD and Cup segmentation
and notching information.

* Reference: [DRISHTIGS1-2014]_
* Original resolution (height x width): varying (min: 1749 x 2045, max: 1845 x
  2468)
* Configuration resolution: 1760 x 2048 (after center cropping)
* Test samples: 51
* Split reference: [DRISHTIGS1-2014]_
"""

from bob.db.drishtigs1 import Database as DRISHTI
from bob.ip.binseg.data.transforms import CenterCrop
from bob.ip.binseg.data.binsegdataset import BinSegDataset

#### Config ####

_transforms = [CenterCrop((1760, 2048))]

# bob.db.dataset init
bobdb = DRISHTI(protocol="default_cup")

# PyTorch dataset
dataset = BinSegDataset(bobdb, split="test", transforms=_transforms)
