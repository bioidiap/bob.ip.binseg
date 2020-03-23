#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""REFUGE (training set) for Cup Segmentation

The dataset consists of 1200 color fundus photographs, created for a MICCAI
challenge. The goal of the challenge is to evaluate and compare automated
algorithms for glaucoma detection and optic disc/cup segmentation on a common
dataset of retinal fundus images.

* Reference: [REFUGE-2018]_
* Original resolution (height x width): 2056 x 2124
* Configuration resolution: 1632 x 1632 (after center cropping)
* Training samples: 400
* Split reference: [REFUGE-2018]_

.. warning:

   Notice that the data cropping/resizing algorithm applied on training and
   "validation" data are slightly different and need to be cross-checked.

"""

from bob.db.refuge import Database as REFUGE
from bob.ip.binseg.data.transforms import *
from bob.ip.binseg.data.binsegdataset import BinSegDataset

#### Config ####

transforms = Compose(
    [
        Resize((1539)),
        Pad((21, 46, 22, 47)),  #(left, top, right, bottom)
        RandomHFlip(),
        RandomVFlip(),
        RandomRotation(),
        ColorJitter(),
        ToTensor(),
    ]
)

# bob.db.dataset init
bobdb = REFUGE(protocol="default_cup")

# PyTorch dataset
dataset = BinSegDataset(bobdb, split="train", transform=transforms)
