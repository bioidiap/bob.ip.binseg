#!/usr/bin/env python
# coding=utf-8

"""IOSTAR (training set) for Vessel and Optic-Disc Segmentation

The IOSTAR vessel segmentation dataset includes 30 images with a resolution of
1024 × 1024 pixels. All the vessels in this dataset are annotated by a group of
experts working in the field of retinal image analysis. Additionally the
dataset includes annotations for the optic disc and the artery/vein ratio.

* Reference: [IOSTAR-2016]_
* Original resolution (height x width): 1024 x 1024
* Split reference: [MEYER-2017]_
* Protocol ``vessel``:

  * Training samples: 20 (including labels and masks)
  * Test samples: 10 (including labels and masks)

* Protocol ``optic-disc``:

  * Training samples: 20 (including labels and masks)
  * Test samples: 10 (including labels and masks)
"""

import os
import pkg_resources

import bob.extension

from ..jsondataset import JSONDataset
from ..loader import load_pil_rgb, load_pil_1

_protocols = [
        pkg_resources.resource_filename(__name__, "vessel.json"),
        pkg_resources.resource_filename(__name__, "optic-disc.json"),
        ]

_root_path = bob.extension.rc.get('bob.ip.binseg.iostar.datadir',
        os.path.realpath(os.curdir))

def _loader(s):
    return dict(
            data=load_pil_rgb(s["data"]),
            label=load_pil_1(s["label"]),
            mask=load_pil_1(s["mask"]),
            )

dataset = JSONDataset(protocols=_protocols, root_path=_root_path, loader=_loader)
"""IOSTAR dataset object"""
