#!/usr/bin/env python

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Tim Laibacher, tim.laibacher@idiap.ch
# SPDX-FileContributor: Oscar Jiménez del Toro, oscar.jimenez@idiap.ch
# SPDX-FileContributor: Maxime Délitroz, maxime.delitroz@idiap.ch
# SPDX-FileContributor: Andre Anjos andre.anjos@idiap.ch
# SPDX-FileContributor: Daniel Carron, daniel.carron@idiap.ch
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""DRIVE cross-evaluation dataset with matched resolution.

* Configuration resolution: 544 x 544
"""

from deepdraw.binseg.configs.datasets.drive.xtest import dataset as _xt
from deepdraw.binseg.configs.datasets.drive.xtest import (  # noqa
    second_annotator,
)
from deepdraw.common.data.transforms import Crop, Pad, Resize

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
    "hrf (train)": _xt["hrf (train)"].copy([Resize(363), Pad((0, 90, 0, 91))]),
    "hrf (test)": _xt["hrf (test)"].copy([Resize(363), Pad((0, 90, 0, 91))]),
    "iostar (train)": _xt["iostar (train)"].copy([Resize(544)]),
    "iostar (test)": _xt["iostar (test)"].copy([Resize(544)]),
}
