#!/usr/bin/env python
# coding=utf-8

"""CHASE-DB1 cross-evaluation dataset with matched resolution

* Configuration resolution (height x width): 960 x 960
"""

from bob.ip.binseg.data.transforms import CenterCrop, Pad, Resize
from bob.ip.binseg.configs.datasets.chasedb1.xtest import dataset as _xt

dataset = {
    "train": _xt["train"],
    "test": _xt["test"],
    "drive": _xt["drive"].copy([CenterCrop((544, 544)), Resize(960)]),
    "stare": _xt["stare"].copy(
        [Pad((0, 32, 0, 32)), Resize(960), CenterCrop(960)]
    ),
    "hrf": _xt["hrf"].copy([Pad((0, 584, 0, 584)), Resize(960)]),
    "iostar": _xt["iostar"].copy([Resize(960)]),
}
