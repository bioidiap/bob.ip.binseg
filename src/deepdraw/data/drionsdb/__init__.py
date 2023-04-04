# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""DRIONS-DB (training set) for Optic Disc Segmentation.

The dataset originates from data collected from 55 patients with glaucoma
(23.1%) and eye hypertension (76.9%), and random selected from an eye fundus
image base belonging to the Ophthalmology Service at Miguel Servet Hospital,
Saragossa (Spain).  It contains 110 eye fundus images with a resolution of 600
x 400. Two sets of ground-truth optic disc annotations are available. The first
set is commonly used for training and testing. The second set acts as a "human"
baseline.

* Reference: [DRIONSDB-2008]_
* Original resolution (height x width): 400 x 600
* Configuration resolution: 416 x 608 (after padding)
* Split reference: [MANINIS-2016]_
* Protocols ``expert1`` (baseline) and ``expert2`` (human comparison):

    * Training samples: 60
    * Test samples: 50
"""

import csv
import os

import PIL.Image
import PIL.ImageDraw
import pkg_resources

from ...data.dataset import JSONDataset
from ...utils.rc import load_rc
from ..loader import load_pil_1, load_pil_rgb, make_delayed

_protocols = [
    pkg_resources.resource_filename(__name__, "expert1.json"),
    pkg_resources.resource_filename(__name__, "expert2.json"),
]

_root_path = load_rc().get("datadir.drionsdb", os.path.realpath(os.curdir))

_pkg_path = pkg_resources.resource_filename(__name__, "masks")


def _txt_to_pil_1(fname, size):
    """Converts DRIONS-DB annotations to image format."""
    with open(fname) as f:
        rows = csv.reader(f, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)
        data = list(map(tuple, rows))

    retval = PIL.Image.new("1", size)
    draw = PIL.ImageDraw.ImageDraw(retval)
    draw.polygon(data, fill="white")
    del draw
    return retval


def _pad_right(img):
    """Pads image on the right by one pixel, respects mode."""
    retval = PIL.Image.new(img.mode, (img.size[0] + 1, img.size[1]), "black")
    retval.paste(img, (0, 0) + img.size)  # top-left pasting
    return retval


def _raw_data_loader(sample):
    data = load_pil_rgb(os.path.join(_root_path, sample["data"]))
    label = _txt_to_pil_1(os.path.join(_root_path, sample["label"]), data.size)
    mask = load_pil_1(os.path.join(_pkg_path, sample["mask"]))
    return dict(data=data, label=label, mask=mask)


def _sample_101_loader(sample):
    # pads the image on the right side to account for a difference in
    # resolution to other images in the dataset
    retval = _raw_data_loader(sample)
    retval["data"] = _pad_right(retval["data"])
    retval["label"] = _pad_right(retval["label"])
    retval["mask"] = _pad_right(retval["mask"])
    return retval


def _loader(context, sample):
    if sample["data"].endswith("_101.jpg"):
        return make_delayed(sample, _sample_101_loader)
    return make_delayed(sample, _raw_data_loader)


dataset = JSONDataset(
    protocols=_protocols, fieldnames=("data", "label", "mask"), loader=_loader
)
"""DRIONSDB dataset object."""
