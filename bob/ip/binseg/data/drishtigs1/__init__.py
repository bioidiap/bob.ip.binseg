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

from ..jsondataset import JSONDataset
from ..loader import load_pil_rgb, load_pil_1

_protocols = {
        "optic-disc-all": pkg_resources.resource_filename(__name__, "optic-disc.json"),
        "optic-cup-all": pkg_resources.resource_filename(__name__, "optic-cup.json"),
        "optic-disc-any": pkg_resources.resource_filename(__name__, "optic-disc.json"),
        "optic-cup-any": pkg_resources.resource_filename(__name__, "optic-cup.json"),
        }

_root_path = bob.extension.rc.get('bob.ip.binseg.drishtigs1.datadir',
        os.path.realpath(os.curdir))

def _loader(context, sample):
    retval = dict(
            data=load_pil_rgb(sample["data"]),
            label=load_pil_rgb(sample["label"]).convert("L"),
            )
    # Drishti-GS provides softmaps of multiple annotators
    # we threshold to get gt where all/any of the annotators overlap
    if context["protocol"].endswith("-all"):
        retval["label"] = retval["label"].point(lambda p: p>254, mode="1")
    elif context["protocol"].endswith("-any"):
        retval["label"] = retval["label"].point(lambda p: p>0, mode="1")
    else:
        raise RuntimeError(f"Unknown protocol {context['protocol']}")
    return retval

dataset = JSONDataset(protocols=_protocols, root_path=_root_path, loader=_loader)
"""Drishti-GS1 dataset object"""
