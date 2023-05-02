# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""HRF cross-evaluation dataset with matched resolution.

* Configuration resolution: 1168 x 1648
"""

from ....data.transforms import Crop, Pad, Resize
from .xtest import dataset as _xt

dataset = {
    "train": _xt["train"],
    "test": _xt["test"],
    "train (full resolution)": _xt["train (full resolution)"],
    "test (full resolution)": _xt["test (full resolution)"],
    "drive (train)": _xt["drive (train)"].copy(
        [Crop(75, 10, 416, 544), Pad((21, 0, 22, 0)), Resize(1168)]
    ),
    "drive (test)": _xt["drive (test)"].copy(
        [Crop(75, 10, 416, 544), Pad((21, 0, 22, 0)), Resize(1168)]
    ),
    "stare (train)": _xt["stare (train)"].copy(
        [Crop(50, 0, 500, 705), Resize(1168), Pad((1, 0, 1, 0))]
    ),
    "stare (test)": _xt["stare (test)"].copy(
        [Crop(50, 0, 500, 705), Resize(1168), Pad((1, 0, 1, 0))]
    ),
    "chasedb1 (train)": _xt["chasedb1 (train)"].copy(
        [Crop(140, 18, 680, 960), Resize(1168)]
    ),
    "chasedb1 (test)": _xt["chasedb1 (test)"].copy(
        [Crop(140, 18, 680, 960), Resize(1168)]
    ),
    "iostar (train)": _xt["iostar (train)"].copy(
        [Crop(144, 0, 768, 1024), Pad((30, 0, 30, 0)), Resize(1168)]
    ),
    "iostar (test)": _xt["iostar (test)"].copy(
        [Crop(144, 0, 768, 1024), Pad((30, 0, 30, 0)), Resize(1168)]
    ),
}
