# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""CXR8  (Cropped around Ground-Truth) cross-evaluation dataset."""

from ..jsrt.default_gtcrop import dataset as _jsrt
from ..montgomery.default_gtcrop import dataset as _mc
from ..shenzhen.default_gtcrop import dataset as _shenzhen
from .default_gtcrop import dataset as _cxr8

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
