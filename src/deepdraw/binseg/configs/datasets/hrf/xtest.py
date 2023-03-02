# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""HRF cross-evaluation dataset."""

from ..chasedb1.first_annotator import dataset as _chase
from ..drive.default import dataset as _drive
from ..iostar.vessel import dataset as _iostar
from ..stare.ah import dataset as _stare
from .default import dataset as _hrf

dataset = {
    "train": _hrf["train"],
    "test": _hrf["test"],
    "train (full resolution)": _hrf["train (full resolution)"],
    "test (full resolution)": _hrf["test (full resolution)"],
    "drive (train)": _drive["train"],
    "drive (test)": _drive["test"],
    "stare (train)": _stare["train"],
    "stare (test)": _stare["test"],
    "chasedb1 (train)": _chase["train"],
    "chasedb1 (test)": _chase["test"],
    "iostar (train)": _iostar["train"],
    "iostar (test)": _iostar["test"],
}
