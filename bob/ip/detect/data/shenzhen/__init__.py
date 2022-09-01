#!/usr/bin/env python
# coding=utf-8

"""Shenzhen No.3 People’s Hospital dataset for Lung Segmentation

The database includes 336 cases with manifestation of tuberculosis, and 326
normal cases.  It contains a total  of 662 images. Image size varies for each
X-ray. It is approximately 3K x 3K. One set of ground-truth lung annotations is
available for 566 of the 662 images.

* Reference: [SHENZHEN-2014]_
* Original resolution (height x width): Approximately 3K x 3K (varies)
* Configuration resolution: 512 x 512 (after rescaling)
* Split reference: [GAAL-2020]_
* Protocol ``default``:

  * Training samples: 396 (including labels)
  * Validation samples: 56 (including labels)
  * Test samples: 114 (including labels)

"""

import os

import pkg_resources

import bob.extension

from ..dataset import JSONDataset
from ..loader import load_pil_1, load_pil_rgb, make_delayed

_protocols = [
    pkg_resources.resource_filename(__name__, "default.json"),
]

_root_path = bob.extension.rc.get(
    "bob.ip.binseg.shenzhen.datadir", os.path.realpath(os.curdir)
)


def _raw_data_loader(sample):
    return dict(
        data=load_pil_rgb(os.path.join(_root_path, sample["data"])),
        label=load_pil_1(os.path.join(_root_path, sample["label"])),
    )


def _loader(context, sample):
    # "context" is ignored in this case - database is homogeneous
    # we returned delayed samples to avoid loading all images at once
    return make_delayed(sample, _raw_data_loader)


dataset = JSONDataset(
    protocols=_protocols, fieldnames=("data", "label"), loader=_loader
)

"""Shenzhen CXR dataset object"""
