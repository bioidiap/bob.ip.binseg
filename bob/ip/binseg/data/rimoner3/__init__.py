#!/usr/bin/env python
# coding=utf-8

"""RIM-ONE r3 (training set) for Cup Segmentation

The dataset contains 159 stereo eye fundus images with a resolution of 2144 x
1424. The right part of the stereo image is disregarded. Two sets of
ground-truths for optic disc and optic cup are available. The first set is
commonly used for training and testing. The second set acts as a “human”
baseline.  A third set, composed of annotation averages may also be used for
training and evaluation purposes.

* Reference: [RIMONER3-2015]_
* Original resolution (height x width): 1424 x 1072
* Split reference: [MANINIS-2016]_
* Protocols ``optic-disc-exp1``, ``optic-cup-exp1``, ``optic-disc-exp2``,
  ``optic-cup-exp2``, ``optic-disc-avg`` and ``optic-cup-avg``:

  * Training: 99
  * Test: 60
"""

import os
import pkg_resources

import bob.extension

from ..jsondataset import JSONDataset
from ..loader import load_pil_rgb, load_pil_1

_protocols = [
        pkg_resources.resource_filename(__name__, "optic-disc-exp1.json"),
        pkg_resources.resource_filename(__name__, "optic-cup-exp1.json"),
        pkg_resources.resource_filename(__name__, "optic-disc-exp2.json"),
        pkg_resources.resource_filename(__name__, "optic-cup-exp2.json"),
        pkg_resources.resource_filename(__name__, "optic-disc-avg.json"),
        pkg_resources.resource_filename(__name__, "optic-cup-avg.json"),
        ]

_root_path = bob.extension.rc.get('bob.ip.binseg.rimoner3.datadir',
        os.path.realpath(os.curdir))

def _loader(context, sample):
    # RIM-ONE r3 provides stereo images - we clip them here to get only the
    # left part of the image, which is also annotated
    return dict(
            data=load_pil_rgb(sample["data"]).crop((0, 0, 1072, 1424)),
            label=load_pil_1(sample["label"]).crop((0, 0, 1072, 1424)),
            )

dataset = JSONDataset(protocols=_protocols, root_path=_root_path, loader=_loader)
"""RIM-ONE r3 dataset object"""
