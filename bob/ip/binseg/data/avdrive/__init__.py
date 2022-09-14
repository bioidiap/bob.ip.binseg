#!/usr/bin/env python
# coding=utf-8

"""AV_DRIVE dataset for Vessel Segmentation

The DRIVE database has been established to enable comparative studies on
segmentation of blood vessels in retinal images.

* Reference: [DRIVE-2004]_
* Original resolution (height x width): 584 x 565
* Split reference: [DRIVE-2004]_
* Protocol ``default``:

  * Training samples: 20 (including labels and masks)
  * Test samples: 20 (including labels from annotator 1 and masks

"""

import os

import pkg_resources

import bob.extension

from ....common.data.dataset import JSONDataset
from ....common.data.loader import load_pil_1, load_pil_rgb, make_delayed

_protocols = [pkg_resources.resource_filename(__name__, "default.json")]

_root_path_drive = bob.extension.rc.get(
    "bob.ip.binseg.drive.datadir", os.path.realpath(os.curdir)
)

_root_path_av_drive = bob.extension.rc.get(
    "bob.ip.binseg.avdrive.datadir", os.path.realpath(os.curdir)
)


def _raw_data_loader(sample):
    return dict(
        data=load_pil_rgb(os.path.join(_root_path_drive, sample["data"])),
        label=load_pil_1(os.path.join(_root_path_av_drive, sample["label"])),
        mask=load_pil_1(os.path.join(_root_path_drive, sample["mask"])),
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
"""DRIVE dataset object"""
