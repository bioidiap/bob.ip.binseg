# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

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

* Protocol ``idiap``:

  * Exactly the same as ``default``, except it uses the file organisation
    suitable to the Idiap Research Institute (where there is limit of 1k files
    per directory)

"""

import os

import numpy as np
import pkg_resources

from PIL import Image

from ...data.dataset import JSONDataset
from ...utils.rc import load_rc
from ..loader import load_pil_rgb, make_delayed

_protocols = [
    pkg_resources.resource_filename(__name__, "default.json"),
]

_rc = load_rc()
_root_path = _rc.get("datadir.cxr8", os.path.realpath(os.curdir))
_idiap_organisation = True
if os.path.exists(os.path.join(_root_path, "images", "00000001_000.png")):
    _idiap_organisation = False


def _raw_data_loader(sample):
    sample_parts = sample["data"].split("/", 1)
    sample_path = (
        os.path.join(sample_parts[0], sample_parts[1][:5], sample_parts[1])
        if _idiap_organisation
        else sample["data"]
    )
    label_parts = sample["data"].split("/", 1)
    label_path = (
        os.path.join(label_parts[0], label_parts[1][:5], label_parts[1])
        if _idiap_organisation
        else sample["label"]
    )
    retval = dict(
        data=load_pil_rgb(os.path.join(_root_path, sample_path)),
        label=np.array(Image.open(os.path.join(_root_path, label_path))),
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
