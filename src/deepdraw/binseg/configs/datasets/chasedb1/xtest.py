# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""CHASE-DB1 cross-evaluation dataset."""

from ..drive.default import dataset as _drive
from ..hrf.default import dataset as _hrf
from ..iostar.vessel import dataset as _iostar
from ..stare.ah import dataset as _stare
from .first_annotator import dataset as _chase
from .first_annotator import second_annotator  # noqa

dataset = {
    "train": _chase["train"],
    "test": _chase["test"],
    "drive (train)": _drive["train"],
    "drive (test)": _drive["test"],
    "stare (train)": _stare["train"],
    "stare (test)": _stare["test"],
    "hrf (train)": _hrf["train"],
    "hrf (test)": _hrf["test"],
    "iostar (train)": _iostar["train"],
    "iostar (test)": _iostar["test"],
}
