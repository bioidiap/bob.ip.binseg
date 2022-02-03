#!/usr/bin/env python
# coding=utf-8

"""ChestX-ray8: Hospital-scale Chest X-ray Database

The database contains a total  of 112120 images. Image size for each X-ray is
1024 x 1024. One set of mask annotations is available for all images.

* Reference: [CXR8-2017]_
* Original resolution (height x width): 1024 x 1024
* Configuration resolution: 256 x 256 (after rescaling)
* Split reference: [GAAL-2020]_
* Protocol ``default``:

  * Training samples: 78484 (including labels)
  * Validation samples: 11212 (including labels)
  * Test samples: 22424 (including labels)

"""

import os

import numpy as np
import pkg_resources

from PIL import Image

import bob.extension

from ..dataset import JSONDataset
from ..loader import load_pil_rgb, make_delayed

_protocols = [
    pkg_resources.resource_filename(__name__, "default.json"),
]

_root_path = bob.extension.rc.get(
    "bob.ip.binseg.cxr8.datadir", os.path.realpath(os.curdir)
)


def _raw_data_loader(sample):
    retval = dict(
        data=load_pil_rgb(os.path.join(_root_path, sample["data"])),
        label=np.array(Image.open(os.path.join(_root_path, sample["label"]))),
    )

    retval["label"] = np.where(retval["label"] == 255, 0, retval["label"])
    retval["label"] = Image.fromarray(np.array(retval["label"] > 0))
    return retval


def _loader(context, sample):
    # "context" is ignored in this case - database is homogeneous
    # we returned delayed samples to avoid loading all images at once
    return make_delayed(sample, _raw_data_loader)


dataset = JSONDataset(
    protocols=_protocols, fieldnames=("data", "label"), loader=_loader
)

"""CXR8 dataset object"""
