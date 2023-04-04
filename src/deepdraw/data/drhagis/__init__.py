# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""DRHAGIS dataset for Vessel Segmentation.

The DR HAGIS database has been created to aid the development of vessel extraction algorithms
suitable for retinal screening programmes. Researchers are encouraged to test their
segmentation algorithms using this database.

It should be noted that image 24 and 32 are identical, as this fundus image was obtained
from a patient exhibiting both diabetic retinopathy and age-related macular degeneration.


The images resolutions (height x width) are:
    - 4752x3168  or
    - 3456x2304  or
    - 3126x2136  or
    - 2896x1944  or
    - 2816x1880  or

* Protocol ``default``:

  * Training samples: 19 (including labels and masks)
  * Test samples: 20 (including labels and masks)
"""

import os

import pkg_resources

from ...data.dataset import JSONDataset
from ...utils.rc import load_rc
from ..loader import load_pil_1, load_pil_rgb, make_delayed

_protocols = [
    pkg_resources.resource_filename(__name__, "default.json"),
]

_root_path = load_rc().get("datadir.drhagis", os.path.realpath(os.curdir))


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
    protocols=_protocols,
    fieldnames=("data", "label", "mask"),
    loader=_loader,
)
"""DRHAGIS dataset object."""
