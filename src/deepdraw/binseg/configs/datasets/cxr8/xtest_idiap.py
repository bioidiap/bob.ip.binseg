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

"""CXR8 cross-evaluation dataset with Idiap directory structure
organisation."""

from bob.ip.binseg.configs.datasets.cxr8.idiap import dataset as _cxr8
from bob.ip.binseg.configs.datasets.jsrt.default import dataset as _jsrt
from bob.ip.binseg.configs.datasets.montgomery.default import dataset as _mc
from bob.ip.binseg.configs.datasets.shenzhen.default import dataset as _shenzhen

dataset = {
    "train": _cxr8["train"],
    "validation": _cxr8["validation"],
    "test": _cxr8["test"],
    "montgomery (train)": _mc["train"],
    "montgomery (validation)": _mc["validation"],
    "montgomery (test)": _mc["test"],
    "jsrt (train)": _jsrt["train"],
    "jsrt (validation)": _jsrt["validation"],
    "jsrt (test)": _jsrt["test"],
    "shenzhen (train)": _shenzhen["train"],
    "shenzhen (validation)": _shenzhen["validation"],
    "shenzhen (test)": _shenzhen["test"],
}
