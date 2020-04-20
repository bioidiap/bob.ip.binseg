#!/usr/bin/env python
# coding=utf-8

"""STARE dataset for Vessel Segmentation

A subset of the original STARE dataset contains 20 annotated eye fundus images
with a resolution of 700 x 605 (width x height). Two sets of ground-truth
vessel annotations are available. The first set by Adam Hoover ("ah") is
commonly used for training and testing. The second set by Valentina Kouznetsova
("vk") is typically used as a “human” baseline.

* Reference: [STARE-2000]_
* Original resolution (width x height): 700 x 605
* Split reference: [MANINIS-2016]_
* Protocol ``ah`` (default baseline):

  * Training samples: 10 (including labels from annotator "ah")
  * Test samples: 10 (including labels from annotator "ah")

* Protocol ``vk`` (normally used as human comparison):

  * Training samples: 10 (including labels from annotator "vk")
  * Test samples: 10 (including labels from annotator "vk")

"""

import os
import pkg_resources

import bob.extension

from ..dataset import JSONDataset
from ..loader import load_pil_rgb, load_pil_1, data_path_keymaker

_protocols = [
    pkg_resources.resource_filename(__name__, "ah.json"),
    pkg_resources.resource_filename(__name__, "vk.json"),
]

_fieldnames = ("data", "label")

_root_path = bob.extension.rc.get(
    "bob.ip.binseg.stare.datadir", os.path.realpath(os.curdir)
)

def _make_loader(root_path):

    def _loader(context, sample):
        # "context" is ignore in this case - database is homogeneous
        return dict(
            data=load_pil_rgb(os.path.join(root_path, sample["data"])),
            label=load_pil_1(os.path.join(root_path, sample["label"])),
        )
    return _loader


dataset = JSONDataset(
    protocols=_protocols,
    fieldnames=_fieldnames,
    loader=_make_loader(_root_path),
    keymaker=data_path_keymaker,
)
"""STARE dataset object"""
