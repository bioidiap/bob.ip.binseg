# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Montgomery County (Cropped Around Ground-Truth) cross-evaluation dataset."""

from ..jsrt.default_gtcrop import dataset as _jsrt
from ..shenzhen.default_gtcrop import dataset as _shenzhen
from .default_gtcrop import dataset as _mc

dataset = {
    "train": _mc["train"],
    "validation": _mc["validation"],
    "test": _mc["test"],
    "jsrt (train)": _jsrt["train"],
    "jsrt (validation)": _jsrt["validation"],
    "jsrt (test)": _jsrt["test"],
    "shenzhen (train)": _shenzhen["train"],
    "shenzhen (validation)": _shenzhen["validation"],
    "shenzhen (test)": _shenzhen["test"],
}
