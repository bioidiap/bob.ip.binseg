#!/usr/bin/env python
# coding=utf-8

"""HRF dataset for Vessel Segmentation

The database includes 15 images of each healthy, diabetic retinopathy (DR), and
glaucomatous eyes.  It contains a total  of 45 eye fundus images with a
resolution of 3304 x 2336. One set of ground-truth vessel annotations is
available.

* Reference: [HRF-2013]_
* Original resolution (height x width): 2336 x 3504
* Configuration resolution: 1168 x 1648 (after specific cropping and rescaling)
* Split reference: [ORLANDO-2017]_
* Protocol ``default``:

  * Training samples: 15 (including labels)
  * Test samples: 30 (including labels)

"""

import os
import pkg_resources

import bob.extension

from ..dataset import JSONDataset
from ..loader import load_pil_rgb, load_pil_1, make_delayed

_protocols = [
    pkg_resources.resource_filename(__name__, "default.json"),
]

_root_path = bob.extension.rc.get(
    "bob.ip.binseg.hrf.datadir", os.path.realpath(os.curdir)
)


def _raw_data_loader(sample):
    return dict(
        data=load_pil_rgb(os.path.join(_root_path, sample["data"])),
        label=load_pil_1(os.path.join(_root_path, sample["label"])),
        mask=load_pil_1(os.path.join(_root_path, sample["mask"])),
    )


def _loader(context, sample):
    # "context" is ignored in this case - database is homogeneous
    # we returned delayed samples to avoid loading all images at once
    return make_delayed(sample, _raw_data_loader)


dataset = JSONDataset(
    protocols=_protocols, fieldnames=("data", "label", "mask"), loader=_loader,
)
"""HRF dataset object"""
