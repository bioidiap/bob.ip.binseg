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

"""CHASE-DB1 cross-evaluation dataset."""

from deepdraw.binseg.configs.datasets.chasedb1.first_annotator import (
    dataset as _chase,
)
from deepdraw.binseg.configs.datasets.chasedb1.first_annotator import (  # noqa
    second_annotator,
)
from deepdraw.binseg.configs.datasets.drive.default import dataset as _drive
from deepdraw.binseg.configs.datasets.hrf.default import dataset as _hrf
from deepdraw.binseg.configs.datasets.iostar.vessel import dataset as _iostar
from deepdraw.binseg.configs.datasets.stare.ah import dataset as _stare

dataset = {
    "train": _chase["train"],
    "test": _chase["test"],
    "drive (train)": _drive["train"],
    "drive (test)": _drive["test"],
    "stare (train)": _stare["train"],
    "stare (test)": _stare["test"],
    "hrf (train)": _hrf["train"],
    "hrf (test)": _hrf["test"],
    "iostar (train)": _iostar["train"],
    "iostar (test)": _iostar["test"],
}