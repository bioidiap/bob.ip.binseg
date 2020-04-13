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

from ..jsondataset import JSONDataset
from ..loader import load_pil_rgb, load_pil_1

_protocols = [
        pkg_resources.resource_filename(__name__, "default.json"),
        ]

_root_path = bob.extension.rc.get('bob.ip.binseg.hrf.datadir',
        os.path.realpath(os.curdir))

def _loader(context, sample):
    #"context" is ignore in this case - database is homogeneous
    return dict(
            data=load_pil_rgb(sample["data"]),
            label=load_pil_1(sample["label"]),
            mask=load_pil_1(sample["mask"]),
            )

dataset = JSONDataset(protocols=_protocols, root_path=_root_path, loader=_loader)
"""HRF dataset object"""
