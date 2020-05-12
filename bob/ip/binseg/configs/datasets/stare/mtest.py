#!/usr/bin/env python
# coding=utf-8

"""STARE cross-evaluation dataset with matched resolution

* Configuration resolution: 704 x 608
"""

from bob.ip.binseg.data.transforms import CenterCrop, Pad, Resize
from bob.ip.binseg.configs.datasets.stare.xtest import (
    dataset as _xt,
    second_annotator,
)

dataset = {
    "train": _xt["train"],
    "test": _xt["test"],
    "drive (train)": _xt["drive (train)"].copy(
        [CenterCrop((470, 544)), Pad((10, 9, 10, 8)), Resize(608)]
    ),
    "drive (test)": _xt["drive (test)"].copy(
        [CenterCrop((470, 544)), Pad((10, 9, 10, 8)), Resize(608)]
    ),
    "chasedb1 (train)": _xt["chasedb1 (train)"].copy(
        [CenterCrop((829, 960)), Resize(608)]
    ),
    "chasedb1 (test)": _xt["chasedb1 (test)"].copy(
        [CenterCrop((829, 960)), Resize(608)]
    ),
    "hrf (train)": _xt["hrf (train)"].copy(
        [Pad((0, 345, 0, 345)), Resize(608)]
    ),
    "hrf (test)": _xt["hrf (test)"].copy([Pad((0, 345, 0, 345)), Resize(608)]),
    "iostar (train)": _xt["iostar (train)"].copy(
        [Pad((81, 0, 81, 0)), Resize(608)]
    ),
    "iostar (test)": _xt["iostar (test)"].copy(
        [Pad((81, 0, 81, 0)), Resize(608)]
    ),
}
