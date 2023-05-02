# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""STARE cross-evaluation dataset."""

from ..chasedb1.first_annotator import dataset as _chase
from ..drive.default import dataset as _drive
from ..hrf.default import dataset as _hrf
from ..iostar.vessel import dataset as _iostar
from .ah import dataset as _stare
from .ah import second_annotator  # noqa

dataset = {
    "train": _stare["train"],
    "test": _stare["test"],
    "drive (train)": _drive["train"],
    "drive (test)": _drive["test"],
    "chasedb1 (train)": _chase["train"],
    "chasedb1 (test)": _chase["test"],
    "hrf (train)": _hrf["train"],
    "hrf (test)": _hrf["test"],
    "iostar (train)": _iostar["train"],
    "iostar (test)": _iostar["test"],
}
