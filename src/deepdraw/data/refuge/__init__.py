# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""REFUGE for Optic Disc and Cup Segmentation.

The dataset consists of 1200 color fundus photographs, created for a MICCAI
challenge. The goal of the challenge is to evaluate and compare automated
algorithms for glaucoma detection and optic disc/cup segmentation on a common
dataset of retinal fundus images.

* Reference (including train/dev/test split): [REFUGE-2018]_
* Protocols ``optic-disc`` and ``cup``:

  * Training samples:

    * 400
    * includes optic-disc and cup labels
    * includes label: glaucomatous and non-glaucomatous
    * original resolution: 2056 x 2124

  * Validation samples:

    * 400
    * includes optic-disc and cup labels
    * original resolution: 1634 x 1634

  * Test samples:

    * 400
    * includes optic-disc and cup labels
    * includes label: glaucomatous and non-glaucomatous
    * original resolution:
"""

import os

import pkg_resources

from ...data.dataset import JSONDataset
from ...utils.rc import load_rc
from ..loader import load_pil_1, load_pil_rgb, make_delayed

_protocols = {
    "optic-disc": pkg_resources.resource_filename(__name__, "default.json"),
    "optic-cup": pkg_resources.resource_filename(__name__, "default.json"),
}

_root_path = load_rc().get("datadir.refuge", os.path.realpath(os.curdir))
_pkg_path = pkg_resources.resource_filename(__name__, "masks")


def _disc_loader(sample):
    retval = dict(
        data=load_pil_rgb(os.path.join(_root_path, sample["data"])),
        label=load_pil_rgb(os.path.join(_root_path, sample["label"])),
        mask=load_pil_1(os.path.join(_pkg_path, sample["mask"])),
    )
    if "glaucoma" in sample:
        retval["glaucoma"] = sample["glaucoma"]
    retval["label"] = retval["label"].convert("L")
    retval["label"] = retval["label"].point(lambda p: p <= 150, mode="1")
    return retval


def _cup_loader(sample):
    retval = dict(
        data=load_pil_rgb(os.path.join(_root_path, sample["data"])),
        label=load_pil_rgb(os.path.join(_root_path, sample["label"])),
        mask=load_pil_1(os.path.join(_pkg_path, sample["mask"])),
    )
    if "glaucoma" in sample:
        retval["glaucoma"] = sample["glaucoma"]
    retval["label"] = retval["label"].convert("L")
    retval["label"] = retval["label"].point(lambda p: p <= 100, mode="1")
    return retval


def _loader(context, sample):
    if context["subset"] == "train":
        # adds binary metadata for glaucoma/non-glaucoma patients
        sample["glaucoma"] = os.path.basename(sample["label"]).startswith("g")
    elif context["subset"] == "test":
        sample["glaucoma"] = sample["label"].split(os.sep)[-2] == "G"
    elif context["subset"] == "validation":
        pass
    else:
        raise RuntimeError(f"Unknown subset {context['subset']}")

    # optic disc is drawn with gray == 128 and includes the cup, drawn with
    # black == 0.  The rest is white == 255.
    if context["protocol"] == "optic-disc":
        return make_delayed(sample, _disc_loader)
    elif context["protocol"] == "optic-cup":
        return make_delayed(sample, _cup_loader)
    else:
        raise RuntimeError(f"Unknown protocol {context['protocol']}")


dataset = JSONDataset(
    protocols=_protocols,
    fieldnames=("data", "label", "mask"),
    loader=_loader,
)
"""REFUGE dataset object."""
