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

_root_path = bob.extension.rc.get('bob.db.drive.datadir',
        os.path.realpath(os.curdir))

def _loader(s):
    return dict(
            data=load_pil_rgb(s["data"]),
            label=load_pil_1(s["label"]),
            mask=load_pil_1(s["mask"]),
            )

dataset = JSONDataset(protocols=_protocols, root_path=_root_path, loader=_loader)
"""DRIVE dataset for Vessel Segmentation

The DRIVE database has been established to enable comparative studies on
segmentation of blood vessels in retinal images.

* Reference: [DRIVE-2004]_
* Original resolution (height x width): 584 x 565
* Training samples: 20 (including labels and masks)
* Test samples: 20 (including labels from 2 annotators and masks)
* Split reference: [DRIVE-2004]_
"""
