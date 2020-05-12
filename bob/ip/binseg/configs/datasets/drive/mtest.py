#!/usr/bin/env python
# coding=utf-8

"""DRIVE cross-evaluation dataset with matched resolution

* Configuration resolution: 544 x 544
"""

from bob.ip.binseg.data.transforms import Resize, Pad, Crop
from bob.ip.binseg.configs.datasets.drive.xtest import (
    dataset as _xt,
    second_annotator,
)

dataset = {
    "train": _xt["train"],
    "test": _xt["test"],
    "stare (train)": _xt["stare (train)"].copy(
        [Resize(471), Pad((0, 37, 0, 36))]
    ),
    "stare (test)": _xt["stare (test)"].copy(
        [Resize(471), Pad((0, 37, 0, 36))]
    ),
    "chasedb1 (train)": _xt["chasedb1 (train)"].copy(
        [Resize(544), Crop(0, 12, 544, 544)]
    ),
    "chasedb1 (test)": _xt["chasedb1 (test)"].copy(
        [Resize(544), Crop(0, 12, 544, 544)]
    ),
    "hrf (train)": _xt["hrf (train)"].copy([Resize((363)), Pad((0, 90, 0, 91))]),
    "hrf (test)": _xt["hrf (test)"].copy([Resize((363)), Pad((0, 90, 0, 91))]),
    "iostar (train)": _xt["iostar (train)"].copy([Resize(544)]),
    "iostar (test)": _xt["iostar (test)"].copy([Resize(544)]),
}
