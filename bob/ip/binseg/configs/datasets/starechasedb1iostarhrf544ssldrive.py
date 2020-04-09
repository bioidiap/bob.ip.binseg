#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""DRIVE (SSL training set) for Vessel Segmentation

The DRIVE database has been established to enable comparative studies on
segmentation of blood vessels in retinal images.

* Reference: [DRIVE-2004]_
* Configuration resolution: 544 x 544

The dataset available in this file is composed of STARE, CHASE-DB1, IOSTAR
vessel and HRF (with annotated samples) and DRIVE without labels.
"""

# Labelled bits
import torch.utils.data
from bob.ip.binseg.configs.datasets.stare544 import dataset as _stare
from bob.ip.binseg.configs.datasets.chasedb1544 import dataset as _chase
from bob.ip.binseg.configs.datasets.iostarvessel544 import dataset as _iostar
from bob.ip.binseg.configs.datasets.hrf544 import dataset as _hrf
_labelled = torch.utils.data.ConcatDataset([_stare, _chase, _iostar, _hrf])

# Use DRIVE without labels in this setup
from .drive import dataset as _unlabelled

from bob.ip.binseg.data.utils import SSLDataset
dataset = SSLDataset(_labelled, _unlabelled)
