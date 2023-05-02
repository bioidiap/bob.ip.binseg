# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""COVD-IOSTAR for Vessel Segmentation.

* Configuration resolution: 1024 x 1024

The dataset available in this file is composed of DRIVE, STARE, CHASE-DB1, and
HRF (with annotated samples).

For details on those datasets, consult:

* See :py:mod:`deepdraw.data.drive`
* See :py:mod:`deepdraw.data.stare`
* See :py:mod:`deepdraw.data.chasedb1`
* See :py:mod:`deepdraw.data.hrf`
"""

from torch.utils.data import ConcatDataset

from .. import augment_subset as _augment
from .vessel import dataset as _baseline
from .vessel_mtest import dataset as _mtest

dataset = dict(**_baseline)
dataset["__train__"] = ConcatDataset(
    [
        _augment(_mtest["drive (train)"], rotation_before=True),
        _augment(_mtest["drive (test)"], rotation_before=True),
        _augment(_mtest["stare (train)"], rotation_before=True),
        _augment(_mtest["stare (test)"], rotation_before=True),
        _augment(_mtest["chasedb1 (train)"], rotation_before=True),
        _augment(_mtest["chasedb1 (test)"], rotation_before=True),
        _augment(_mtest["hrf (train)"], rotation_before=False),
        _augment(_mtest["hrf (test)"], rotation_before=False),
    ]
)
dataset["train"] = ConcatDataset(
    [
        _mtest["drive (train)"],
        _mtest["drive (test)"],
        _mtest["stare (train)"],
        _mtest["stare (test)"],
        _mtest["chasedb1 (train)"],
        _mtest["chasedb1 (test)"],
        _mtest["hrf (train)"],
        _mtest["hrf (test)"],
    ]
)
dataset["__valid__"] = dataset["train"]
