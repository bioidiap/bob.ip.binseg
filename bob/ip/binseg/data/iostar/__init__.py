#!/usr/bin/env python
# coding=utf-8

"""IOSTAR (training set) for Vessel and Optic-Disc Segmentation

The IOSTAR vessel segmentation dataset includes 30 images with a resolution of
1024 × 1024 pixels. All the vessels in this dataset are annotated by a group of
experts working in the field of retinal image analysis. Additionally the
dataset includes annotations for the optic disc and the artery/vein ratio.

* Reference: [IOSTAR-2016]_
* Original resolution (height x width): 1024 x 1024
* Split reference: [MEYER-2017]_
* Protocol ``vessel``:

  * Training samples: 20 (including labels and masks)
  * Test samples: 10 (including labels and masks)

* Protocol ``optic-disc``:

  * Training samples: 20 (including labels and masks)
  * Test samples: 10 (including labels and masks)
"""

import os
import pkg_resources

import bob.extension

from ..dataset import JSONDataset
from ..loader import load_pil_rgb, load_pil_1, make_delayed
from ..utils import invert_mode1_image, subtract_mode1_images

_protocols = [
    pkg_resources.resource_filename(__name__, "vessel.json"),
    pkg_resources.resource_filename(__name__, "optic-disc.json"),
]

_root_path = bob.extension.rc.get(
    "bob.ip.binseg.iostar.datadir", os.path.realpath(os.curdir)
)


def _vessel_loader(sample):
    return dict(
        data=load_pil_rgb(os.path.join(_root_path, sample["data"])),
        label=load_pil_1(os.path.join(_root_path, sample["label"])),
        mask=load_pil_1(os.path.join(_root_path, sample["mask"])),
    )


def _disc_loader(sample):
    # For optic-disc analysis, the label provided by IOSTAR raw data is the
    # "inverted" (negative) label, and does not consider the mask region, which
    # must be subtracted.  We do this special manipulation here.
    data = load_pil_rgb(os.path.join(_root_path, sample["data"]))
    label = load_pil_1(os.path.join(_root_path, sample["label"]))
    mask = load_pil_1(os.path.join(_root_path, sample["mask"]))
    label = subtract_mode1_images(
        invert_mode1_image(label), invert_mode1_image(mask)
    )
    return dict(data=data, label=label, mask=mask)


def _loader(context, sample):
    if context["protocol"] == "optic-disc":
        return make_delayed(sample, _disc_loader)
    elif context["protocol"] == "vessel":
        return make_delayed(sample, _vessel_loader)
    raise RuntimeError(f"Unknown protocol {context['protocol']}")


dataset = JSONDataset(
    protocols=_protocols, fieldnames=("data", "label", "mask"), loader=_loader,
)
"""IOSTAR dataset object"""
