#!/usr/bin/env python
# coding=utf-8

"""DRIVE cross-evaluation dataset with matched resolution

* Configuration resolution: 544 x 544
"""

from bob.ip.binseg.data.transforms import Resize, Pad, Crop
from bob.ip.binseg.configs.datasets.drive.xtest import dataset as _xt

dataset = {
        "train": _xt["train"],
        "test": _xt["test"],
        "stare": _xt["stare"].copy([Resize(471), Pad((0, 37, 0, 36))]),
        "chasedb1": _xt["chasedb1"].copy([Resize(544), Crop(0, 12, 544, 544)]),
        "hrf": _xt["hrf"].copy([Resize((363)), Pad((0, 90, 0, 91))]),
        "iostar": _xt["iostar"].copy([Resize(544)]),
        }
