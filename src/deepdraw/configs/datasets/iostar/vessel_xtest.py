# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""IOSTAR vessel cross-evaluation dataset."""

from ..chasedb1.first_annotator import dataset as _chase
from ..drive.default import dataset as _drive
from ..hrf.default import dataset as _hrf
from ..stare.ah import dataset as _stare
from .vessel import dataset as _iostar

dataset = {
    "train": _iostar["train"],
    "test": _iostar["test"],
    "drive (train)": _drive["train"],
    "drive (test)": _drive["test"],
    "stare (train)": _stare["train"],
    "stare (test)": _stare["test"],
    "chasedb1 (train)": _chase["train"],
    "chasedb1 (test)": _chase["test"],
    "hrf (train)": _hrf["train"],
    "hrf (test)": _hrf["test"],
}
