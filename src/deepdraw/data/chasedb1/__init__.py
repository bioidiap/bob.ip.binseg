# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""CHASE-DB1 dataset for Vessel Segmentation.

The CHASE_DB1 is a retinal vessel reference dataset acquired from multiethnic
school children. This database is a part of the Child Heart and Health Study in
England (CHASE), a cardiovascular health survey in 200 primary schools in
London, Birmingham, and Leicester. The ocular imaging was carried out in
46 schools and demonstrated associations between retinal vessel tortuosity and
early risk factors for cardiovascular disease in over 1000 British primary
school children of different ethnic origin. The retinal images of both of the
eyes of each child were recorded with a hand-held Nidek NM-200-D fundus camera.
The images were captured at 30 degrees FOV camera. The dataset of images are
characterized by having nonuniform back-ground illumination, poor contrast of
blood vessels as compared with the background and wider arteriolars that have a
bright strip running down the centre known as the central vessel reflex.

* Reference: [CHASEDB1-2012]_
* Original resolution (height x width): 960 x 999
* Split reference: [CHASEDB1-2012]_
* Protocol ``first-annotator``:

  * Training samples: 8 (including labels from annotator "1stHO")
  * Test samples: 20 (including labels from annotator "1stHO")

* Protocol ``second-annotator``:

  * Training samples: 8 (including labels from annotator "2ndHO")
  * Test samples: 20 (including labels from annotator "2ndHO")
"""

import os

import pkg_resources

from ...data.dataset import JSONDataset
from ...utils.rc import load_rc
from ..loader import load_pil_1, load_pil_rgb, make_delayed

_protocols = [
    pkg_resources.resource_filename(__name__, "first-annotator.json"),
    pkg_resources.resource_filename(__name__, "second-annotator.json"),
]

_root_path = load_rc().get("datadir.chasedb1", os.path.realpath(os.curdir))

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
    protocols=_protocols, fieldnames=("data", "label", "mask"), loader=_loader
)
"""CHASE-DB1 dataset object."""
