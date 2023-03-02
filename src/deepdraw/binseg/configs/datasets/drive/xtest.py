# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""DRIVE cross-evaluation dataset."""

from ..chasedb1.first_annotator import dataset as _chase
from ..hrf.default import dataset as _hrf
from ..iostar.vessel import dataset as _iostar
from ..stare.ah import dataset as _stare
from .default import dataset as _drive
from .default import second_annotator  # noqa

dataset = {
    "train": _drive["train"],
    "test": _drive["test"],
    "stare (train)": _stare["train"],
    "stare (test)": _stare["test"],
    "chasedb1 (train)": _chase["train"],
    "chasedb1 (test)": _chase["test"],
    "hrf (train)": _hrf["train"],
    "hrf (test)": _hrf["test"],
    "iostar (train)": _iostar["train"],
    "iostar (test)": _iostar["test"],
}
