# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Montgomery County cross-evaluation dataset."""

from deepdraw.detect.configs.datasets.jsrt.default import dataset as _jsrt
from deepdraw.detect.configs.datasets.montgomery.default import dataset as _mc
from deepdraw.detect.configs.datasets.shenzhen.default import (
    dataset as _shenzhen,
)

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
