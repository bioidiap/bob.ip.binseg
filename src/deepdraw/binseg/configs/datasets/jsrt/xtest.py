# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""JSRT CXR cross-evaluation dataset."""

from ..montgomery.default import dataset as _mc
from ..shenzhen.default import dataset as _shenzhen
from .default import dataset as _jsrt

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
