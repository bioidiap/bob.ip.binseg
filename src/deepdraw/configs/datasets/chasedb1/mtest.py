# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""CHASE-DB1 cross-evaluation dataset with matched resolution.

* Configuration resolution (height x width): 960 x 960
"""

from ....data.transforms import CenterCrop, Pad, Resize
from .xtest import dataset as _xt
from .xtest import second_annotator  # noqa

dataset = {
    "train": _xt["train"],
    "test": _xt["test"],
    "drive (train)": _xt["drive (train)"].copy(
        [CenterCrop((544, 544)), Resize(960)]
    ),
    "drive (test)": _xt["drive (test)"].copy(
        [CenterCrop((544, 544)), Resize(960)]
    ),
    "stare (train)": _xt["stare (train)"].copy(
        [Pad((0, 32, 0, 32)), Resize(960), CenterCrop(960)]
    ),
    "stare (test)": _xt["stare (test)"].copy(
        [Pad((0, 32, 0, 32)), Resize(960), CenterCrop(960)]
    ),
    "hrf (train)": _xt["hrf (train)"].copy(
        [Pad((0, 584, 0, 584)), Resize(960)]
    ),
    "hrf (test)": _xt["hrf (test)"].copy([Pad((0, 584, 0, 584)), Resize(960)]),
    "iostar (train)": _xt["iostar (train)"].copy([Resize(960)]),
    "iostar (test)": _xt["iostar (test)"].copy([Resize(960)]),
}
