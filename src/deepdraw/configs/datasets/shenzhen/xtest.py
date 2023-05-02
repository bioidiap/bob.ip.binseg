# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shenzhen cross-evaluation dataset."""

from ..jsrt.default import dataset as _jsrt
from ..montgomery.default import dataset as _mc
from .default import dataset as _shenzhen

dataset = {
    "train": _shenzhen["train"],
    "validation": _shenzhen["validation"],
    "test": _shenzhen["test"],
    "montgomery (train)": _mc["train"],
    "montgomery (validation)": _mc["validation"],
    "montgomery (test)": _mc["test"],
    "jsrt (train)": _jsrt["train"],
    "jsrt (validation)": _jsrt["validation"],
    "jsrt (test)": _jsrt["test"],
}
