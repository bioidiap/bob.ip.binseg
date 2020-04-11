#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""REFUGE (validation set) for Cup Segmentation

The dataset consists of 1200 color fundus photographs, created for a MICCAI
challenge. The goal of the challenge is to evaluate and compare automated
algorithms for glaucoma detection and optic disc/cup segmentation on a common
dataset of retinal fundus images.

* Reference: [REFUGE-2018]_
* Original resolution (height x width): 2056 x 2124
* Configuration resolution: 1632 x 1632 (after center cropping)
* Validation samples: 400
* Split reference: [REFUGE-2018]_

.. warning:

   Notice 2 aspects before using these configurations:

   1. The data cropping/resizing algorithm applied on training and "validation"
      data are slightly different and need to be cross-checked.
   2. This is the **validation** set!  The real **test** set is still not
      integrated to the originating bob.db.refuge package: See
      https://gitlab.idiap.ch/bob/bob.db.refuge/issues/1

"""

from bob.ip.binseg.data.transforms import CenterCrop
_transforms = [CenterCrop(1632)]

from bob.db.refuge import Database as REFUGE
bobdb = REFUGE(protocol="default_cup")

from bob.ip.binseg.data.binsegdataset import BinSegDataset
dataset = BinSegDataset(bobdb, split="test", transforms=_transforms)
