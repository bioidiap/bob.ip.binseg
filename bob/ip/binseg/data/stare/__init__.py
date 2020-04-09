#!/usr/bin/env python
# coding=utf-8

import os
import pkg_resources

import bob.extension

from ..jsondataset import JSONDataset
from ..loader import load_pil_rgb, load_pil_1

_protocols = [
        pkg_resources.resource_filename(__name__, "default.json"),
        pkg_resources.resource_filename(__name__, "second-annotation.json"),
        ]

_root_path = bob.extension.rc.get('bob.db.stare.datadir',
        os.path.realpath(os.curdir))

def _loader(s):
    return dict(
            data=load_pil_rgb(s["data"]),
            label=load_pil_1(s["label"]),
            )

dataset = JSONDataset(protocols=_protocols, root_path=_root_path, loader=_loader)
"""STARE (training set) for Vessel Segmentation

A subset of the original STARE dataset contains 20 annotated eye fundus images
with a resolution of 700 x 605 (width x height). Two sets of ground-truth
vessel annotations are available. The first set by Adam Hoover is commonly used
for training and testing. The second set by Valentina Kouznetsova acts as a
“human” baseline.

* Reference: [STARE-2000]_
* Original resolution (width x height): 700 x 605
* Training samples: 10
* Test samples: 10
* Samples include labels from 2 annotators (AH, default and VK, seconda
  annotator)
* Split reference: [MANINIS-2016]_
"""
