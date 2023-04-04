# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""STARE dataset for Vessel Segmentation.

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

from ...data.dataset import JSONDataset
from ...utils.rc import load_rc
from ..loader import load_pil_1, load_pil_rgb, make_delayed

_protocols = [
    pkg_resources.resource_filename(__name__, "ah.json"),
    pkg_resources.resource_filename(__name__, "vk.json"),
]

_root_path = load_rc().get("datadir.stare", os.path.realpath(os.curdir))
_pkg_path = pkg_resources.resource_filename(__name__, "masks")


def _raw_data_loader(sample):
    return dict(
        data=load_pil_rgb(os.path.join(_root_path, sample["data"])),
        label=load_pil_1(os.path.join(_root_path, sample["label"])),
        mask=load_pil_1(os.path.join(_pkg_path, sample["mask"])),
    )


def _loader(context, sample):
    # "context" is ignored in this case - database is homogeneous
    # we returned delayed samples to avoid loading all images at once
    return make_delayed(sample, _raw_data_loader)


dataset = JSONDataset(
    protocols=_protocols,
    fieldnames=("data", "label", "mask"),
    loader=_loader,
)
"""STARE dataset object."""
