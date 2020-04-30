#!/usr/bin/env python
# coding=utf-8

"""HRF cross-evaluation dataset with matched resolution

* Configuration resolution: 1168 x 1648
"""

from bob.ip.binseg.data.transforms import Crop, Pad, Resize
from bob.ip.binseg.configs.datasets.hrf.xtest import dataset as _xt

dataset = {
    "train": _xt["train"],
    "test": _xt["test"],
    "drive": _xt["drive"].copy(
        [Crop(75, 10, 416, 544), Pad((21, 0, 22, 0)), Resize(1168)]
    ),
    "stare": _xt["stare"].copy(
        [Crop(50, 0, 500, 705), Resize(1168), Pad((1, 0, 1, 0))]
    ),
    "chasedb1": _xt["chasedb1"].copy([Crop(140, 18, 680, 960), Resize(1168)]),
    "iostar": _xt["iostar"].copy(
        [Crop(144, 0, 768, 1024), Pad((30, 0, 30, 0)), Resize(1168)]
    ),
}
