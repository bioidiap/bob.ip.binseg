#!/usr/bin/env python

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Tim Laibacher, tim.laibacher@idiap.ch
# SPDX-FileContributor: Oscar Jiménez del Toro, oscar.jimenez@idiap.ch
# SPDX-FileContributor: Maxime Délitroz, maxime.delitroz@idiap.ch
# SPDX-FileContributor: Andre Anjos andre.anjos@idiap.ch
# SPDX-FileContributor: Daniel Carron, daniel.carron@idiap.ch
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""CheXphoto dataset for Lung Detection.

The database includes ___. One set of bounding box annotations is available.

* Reference: [XXXX-XXXX]_
* Original resolution (height x width):
* Configuration resolution: 256 x 256 (after rescaling)
* Split reference: [GAAL-2020]_
* Protocol ``default``:

  * Training samples: ___ (including labels)
  * Validation samples: ___ (including labels)
  * Test samples: ___ (including labels)
"""

import os

import pkg_resources

from PIL import Image, ImageDraw

from ....common.data.dataset import JSONDataset
from ....common.data.loader import load_pil_rgb, make_delayed
from ....common.utils.rc import load_rc

_protocols = [
    pkg_resources.resource_filename(__name__, "default.json"),
]

_root_path = load_rc().get("datadir.chexphoto", os.path.realpath(os.curdir))


def _raw_data_loader(sample):
    im = load_pil_rgb(os.path.join(_root_path, sample["data"]))
    width, height = im.size
    bbox = sample["label"]
    shape = [
        (int(width * bbox[0]), int(height * bbox[1])),
        (int(width * bbox[2]), int(height * bbox[3])),
    ]

    mask = Image.new("L", (width, height))
    img1 = ImageDraw.Draw(mask)
    img1.rectangle(shape, fill="white", outline="white", width=0)
    mask = mask.convert(mode="1", dither=None)

    return dict(
        data=im,
        label=mask,
    )


def _loader(context, sample):
    # "context" is ignored in this case - database is homogeneous
    # we returned delayed samples to avoid loading all images at once
    return make_delayed(sample, _raw_data_loader)


dataset = JSONDataset(
    protocols=_protocols,
    fieldnames=("data", "label"),
    loader=_loader,
)

"""CheXphoto dataset object"""
