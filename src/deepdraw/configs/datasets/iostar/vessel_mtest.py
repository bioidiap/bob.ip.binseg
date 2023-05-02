# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""IOSTAR vessel cross-evaluation dataset with matched resolution.

* Configuration resolution: 1024 x 1024
"""

from ....data.transforms import CenterCrop, Crop, Pad, Resize
from .vessel_xtest import dataset as _xt

dataset = {
    "train": _xt["train"],
    "test": _xt["test"],
    "drive (train)": _xt["drive (train)"].copy(
        [CenterCrop((540, 540)), Resize(1024)]
    ),
    "drive (test)": _xt["drive (test)"].copy(
        [CenterCrop((540, 540)), Resize(1024)]
    ),
    "stare (train)": _xt["stare (train)"].copy(
        [Pad((0, 32, 0, 32)), Resize(1024), CenterCrop(1024)]
    ),
    "stare (test)": _xt["stare (test)"].copy(
        [Pad((0, 32, 0, 32)), Resize(1024), CenterCrop(1024)]
    ),
    "chasedb1 (train)": _xt["chasedb1 (train)"].copy(
        [Crop(0, 18, 960, 960), Resize(1024)]
    ),
    "chasedb1 (test)": _xt["chasedb1 (test)"].copy(
        [Crop(0, 18, 960, 960), Resize(1024)]
    ),
    "hrf (train)": _xt["hrf (train)"].copy(
        [Pad((0, 584, 0, 584)), Resize(1024)]
    ),
    "hrf (test)": _xt["hrf (test)"].copy([Pad((0, 584, 0, 584)), Resize(1024)]),
}
