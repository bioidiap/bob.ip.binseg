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

"""COVD-HRF for Vessel Segmentation.

* Configuration resolution: 1168 x 1648

The dataset available in this file is composed of DRIVE STARE, CHASE-DB1, and
IOSTAR vessel (with annotated samples).

For details on those datasets, consult:

* See :py:mod:`deepdraw.binseg.data.drive`
* See :py:mod:`deepdraw.binseg.data.stare`
* See :py:mod:`deepdraw.binseg.data.chasedb1`
* See :py:mod:`deepdraw.binseg.data.iostar`
"""

from torch.utils.data import ConcatDataset

from deepdraw.binseg.configs.datasets import augment_subset as _augment
from deepdraw.binseg.configs.datasets.hrf.default import dataset as _baseline
from deepdraw.binseg.configs.datasets.hrf.mtest import dataset as _mtest

dataset = dict(**_baseline)
dataset["__train__"] = ConcatDataset(
    [
        _augment(_mtest["drive (train)"], rotation_before=True),
        _augment(_mtest["drive (test)"], rotation_before=True),
        _augment(_mtest["stare (train)"], rotation_before=True),
        _augment(_mtest["stare (test)"], rotation_before=True),
        _augment(_mtest["chasedb1 (train)"], rotation_before=True),
        _augment(_mtest["chasedb1 (test)"], rotation_before=True),
        _augment(_mtest["iostar (train)"], rotation_before=True),
        _augment(_mtest["iostar (test)"], rotation_before=True),
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
        _mtest["iostar (train)"],
        _mtest["iostar (test)"],
    ]
)
dataset["__valid__"] = dataset["train"]
