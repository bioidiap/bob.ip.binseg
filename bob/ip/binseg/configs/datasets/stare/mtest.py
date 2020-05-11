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
    "drive": _xt["drive"].copy(
        [CenterCrop((470, 544)), Pad((10, 9, 10, 8)), Resize(608)]
    ),
    "chasedb1": _xt["chasedb1"].copy([CenterCrop((829, 960)), Resize(608)]),
    "hrf": _xt["hrf"].copy([Pad((0, 345, 0, 345)), Resize(608)]),
    "iostar": _xt["iostar"].copy([Pad((81, 0, 81, 0)), Resize(608)]),
}
