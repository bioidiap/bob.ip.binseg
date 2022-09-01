#!/usr/bin/env python
# coding=utf-8

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

import bob.extension

from ..dataset import JSONDataset
from ..loader import load_pil_rgb, make_delayed

_protocols = [
    pkg_resources.resource_filename(__name__, "default.json"),
]

_root_path = bob.extension.rc.get(
    "bob.ip.detect.chexphoto.datadir", os.path.realpath(os.curdir)
)


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
