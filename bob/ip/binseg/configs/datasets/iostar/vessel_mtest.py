#!/usr/bin/env python
# coding=utf-8

"""IOSTAR vessel cross-evaluation dataset with matched resolution

* Configuration resolution: 1024 x 1024
"""

from bob.ip.binseg.data.transforms import CenterCrop, Crop, Pad, Resize
from bob.ip.binseg.configs.datasets.iostar.vessel_xtest import dataset as _xt

dataset = {
    "train": _xt["train"],
    "test": _xt["test"],
    "drive": _xt["drive"].copy([CenterCrop((540, 540)), Resize(1024)]),
    "stare": _xt["stare"].copy(
        [Pad((0, 32, 0, 32)), Resize(1024), CenterCrop(1024)]
    ),
    "chasedb1": _xt["chasedb1"].copy([Crop(0, 18, 960, 960), Resize(1024)]),
    "hrf": _xt["hrf"].copy([Pad((0, 584, 0, 584)), Resize(1024)]),
}
