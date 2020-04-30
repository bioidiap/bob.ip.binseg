#!/usr/bin/env python
# coding=utf-8

"""Drishti-GS1 for Optic Disc and Cup Segmentation

Drishti-GS is a dataset meant for validation of segmenting OD, cup and
detecting notching.  The images in the Drishti-GS dataset have been collected
and annotated by Aravind Eye hospital, Madurai, India. This dataset is of a
single population as all subjects whose eye images are part of this dataset are
Indians.

The dataset is divided into two: a training set and a testing set of images.
Training images (50) are provided with groundtruths for OD and Cup segmentation
and notching information.

* Reference (including train/test split): [DRISHTIGS1-2014]_
* Original resolution (height x width): varying (min: 1749 x 2045, max: 1845 x
  2468)
* Configuration resolution: 1760 x 2048 (after center cropping)
* Protocols ``optic-disc`` and ``optic-cup``:
  * Training: 50
  * Test: 51
"""

import os
import pkg_resources

import bob.extension

from ..dataset import JSONDataset
from ..loader import load_pil_rgb, make_delayed

_protocols = {
    "optic-disc-all": pkg_resources.resource_filename(
        __name__, "optic-disc.json"
    ),
    "optic-cup-all": pkg_resources.resource_filename(
        __name__, "optic-cup.json"
    ),
    "optic-disc-any": pkg_resources.resource_filename(
        __name__, "optic-disc.json"
    ),
    "optic-cup-any": pkg_resources.resource_filename(
        __name__, "optic-cup.json"
    ),
}

_root_path = bob.extension.rc.get(
    "bob.ip.binseg.drishtigs1.datadir", os.path.realpath(os.curdir)
)


def _raw_data_loader_all(sample):
    retval = dict(
        data=load_pil_rgb(os.path.join(_root_path, sample["data"])),
        label=load_pil_rgb(os.path.join(_root_path, sample["label"])).convert(
            "L"
        ),
    )
    retval["label"] = retval["label"].point(lambda p: p > 254, mode="1")
    return retval


def _raw_data_loader_any(sample):
    retval = dict(
        data=load_pil_rgb(os.path.join(_root_path, sample["data"])),
        label=load_pil_rgb(os.path.join(_root_path, sample["label"])).convert(
            "L"
        ),
    )
    retval["label"] = retval["label"].point(lambda p: p > 0, mode="1")
    return retval


def _loader(context, sample):
    # Drishti-GS provides softmaps of multiple annotators
    # we threshold to get gt where all/any of the annotators overlap
    if context["protocol"].endswith("-all"):
        return make_delayed(sample, _raw_data_loader_all)
    elif context["protocol"].endswith("-any"):
        return make_delayed(sample, _raw_data_loader_any)
    else:
        raise RuntimeError(f"Unknown protocol {context['protocol']}")


dataset = JSONDataset(
    protocols=_protocols, fieldnames=("data", "label"), loader=_loader
)
"""Drishti-GS1 dataset object"""
