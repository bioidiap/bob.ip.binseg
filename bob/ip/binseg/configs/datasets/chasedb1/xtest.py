#!/usr/bin/env python
# coding=utf-8

"""CHASE-DB1 cross-evaluation dataset
"""

from bob.ip.binseg.configs.datasets.drive.default import dataset as _drive
from bob.ip.binseg.configs.datasets.stare.ah import dataset as _stare
from bob.ip.binseg.configs.datasets.chasedb1.first_annotator import (
    dataset as _chase,
    second_annotator,
)
from bob.ip.binseg.configs.datasets.hrf.default import dataset as _hrf
from bob.ip.binseg.configs.datasets.iostar.vessel import dataset as _iostar

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
