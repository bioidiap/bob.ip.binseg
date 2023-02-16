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

"""COVD-CHASEDB1 for Vessel Segmentation.

* Configuration resolution (height x width): 960 x 960

The dataset available in this file is composed of DRIVE, STARE, IOSTAR
vessel and HRF (with annotated samples).

For details on those datasets, consult:

* See :py:mod:`deepdraw.binseg.data.drive`
* See :py:mod:`deepdraw.binseg.data.stare`
* See :py:mod:`deepdraw.binseg.data.iostar`
* See :py:mod:`deepdraw.binseg.data.hrf`
"""

from torch.utils.data import ConcatDataset

from deepdraw.binseg.configs.datasets import augment_subset as _augment
from deepdraw.binseg.configs.datasets.chasedb1.first_annotator import (
    dataset as _baseline,
)
from deepdraw.binseg.configs.datasets.chasedb1.first_annotator import (
    second_annotator,
)
from deepdraw.binseg.configs.datasets.chasedb1.mtest import dataset as _mtest

dataset = dict(**_baseline)
dataset["__train__"] = ConcatDataset(
    [
        _augment(_mtest["drive (train)"], rotation_before=True),
        _augment(_mtest["drive (test)"], rotation_before=True),
        _augment(_mtest["stare (train)"], rotation_before=True),
        _augment(_mtest["stare (test)"], rotation_before=True),
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
        _mtest["stare (train)"],
        _mtest["stare (test)"],
        _mtest["hrf (train)"],
        _mtest["hrf (test)"],
        _mtest["iostar (train)"],
        _mtest["iostar (test)"],
    ]
)
dataset["__valid__"] = dataset["train"]
