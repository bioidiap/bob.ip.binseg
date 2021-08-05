#!/usr/bin/env python
# coding=utf-8

"""Japanese Society of Radiological Technology dataset for Lung Segmentation

The database includes 154 nodule and 93 non-nodule images.  It contains a total
of 247 resolution of 2048 x 2048 One set of ground-truth lung annotations is
available.

* Reference: [JSRT-2000]_
* Original resolution (height x width): 2048 x 2048
* Configuration resolution: 512 x 512 (after rescaling)
* Split reference: [GAÁL-2020]_
* Protocol ``default``:

  * Training samples: 172 (including labels)
  * Validation samples: 25 (including labels)
  * Test samples: 50 (including labels)

"""

import os

import numpy as np
import pkg_resources

from PIL import Image

import bob.extension

from ..dataset import JSONDataset
from ..loader import load_pil_1, load_pil_rgb, make_delayed

_protocols = [
    pkg_resources.resource_filename(__name__, "default.json"),
]

_root_path = bob.extension.rc.get(
    "bob.ip.binseg.JSRT.datadir", os.path.realpath(os.curdir)
)


def _raw_data_loader(sample):
    return dict(
        data=load_pil_rgb(os.path.join(_root_path, sample["data"])),
        label=Image.fromarray(
            np.ma.mask_or(
                np.asarray(
                    load_pil_1(os.path.join(_root_path, sample["label_l"]))
                ),
                np.asarray(
                    load_pil_1(os.path.join(_root_path, sample["label_r"]))
                ),
            )
        ),
    )


def _loader(context, sample):
    # "context" is ignored in this case - database is homogeneous
    # we returned delayed samples to avoid loading all images at once
    return make_delayed(sample, _raw_data_loader)


dataset = JSONDataset(
    protocols=_protocols,
    fieldnames=("data", "label_l", "label_r"),
    loader=_loader,
)

"""Japanese Society of Radiological Technology dataset object"""
