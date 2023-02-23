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

"""JSRT CXR cross-evaluation dataset."""

from deepdraw.binseg.configs.datasets.jsrt.default import dataset as _jsrt
from deepdraw.binseg.configs.datasets.montgomery.default import dataset as _mc
from deepdraw.binseg.configs.datasets.shenzhen.default import (
    dataset as _shenzhen,
)

dataset = {
    "train": _jsrt["train"],
    "validation": _jsrt["validation"],
    "test": _jsrt["test"],
    "montgomery (train)": _mc["train"],
    "montgomery (validation)": _mc["validation"],
    "montgomery (test)": _mc["test"],
    "shenzhen (train)": _shenzhen["train"],
    "shenzhen (validation)": _shenzhen["validation"],
    "shenzhen (test)": _shenzhen["test"],
}