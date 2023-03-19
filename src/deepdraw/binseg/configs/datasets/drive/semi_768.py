# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Semi-supervised pretrain dataset for the DRIVE dataset."""

from . import _transform
from .default_768 import dataset as _drive

dataset = {
    "train": _drive["train"],
    "test": _drive["test"],
    "__train__": _transform(_drive["train"]),
    "__valid__": _drive["train"],
}
dataset["__extra_valid__"] = dataset["test"]
