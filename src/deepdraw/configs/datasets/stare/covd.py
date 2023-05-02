# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""COVD-STARE for Vessel Segmentation.

* Configuration resolution: 704 x 608

The dataset available in this file is composed of DRIVE, CHASE-DB1, IOSTAR
vessel and HRF (with annotated samples).

For details on those datasets, consult:

* See :py:mod:`deepdraw.data.drive`
* See :py:mod:`deepdraw.data.chasedb1`
* See :py:mod:`deepdraw.data.iostar`
* See :py:mod:`deepdraw.data.hrf`
"""

from torch.utils.data import ConcatDataset

from .. import augment_subset as _augment
from .ah import dataset as _baseline
from .ah import second_annotator
from .mtest import dataset as _mtest

dataset = dict(**_baseline)
dataset["__train__"] = ConcatDataset(
    [
        _augment(_mtest["drive (train)"], rotation_before=True),
        _augment(_mtest["drive (test)"], rotation_before=True),
        _augment(_mtest["chasedb1 (train)"], rotation_before=True),
        _augment(_mtest["chasedb1 (test)"], rotation_before=True),
        _augment(_mtest["hrf (train)"], rotation_before=False),
        _augment(_mtest["hrf (test)"], rotation_before=False),
        _augment(_mtest["iostar (train)"], rotation_before=False),
        _augment(_mtest["iostar (test)"], rotation_before=False),
    ]
)
del second_annotator["train"]  # mismatch with used train set
dataset["train"] = ConcatDataset(
    [
        _mtest["drive (train)"],
        _mtest["drive (test)"],
        _mtest["chasedb1 (train)"],
        _mtest["chasedb1 (test)"],
        _mtest["hrf (train)"],
        _mtest["hrf (test)"],
        _mtest["iostar (train)"],
        _mtest["iostar (test)"],
    ]
)
dataset["__valid__"] = dataset["train"]
