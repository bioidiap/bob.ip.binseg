#!/usr/bin/env python
# coding=utf-8

"""IOSTAR vessel cross-evaluation dataset
"""

from bob.ip.binseg.configs.datasets.drive.default import dataset as _drive
from bob.ip.binseg.configs.datasets.stare.ah import dataset as _stare
from bob.ip.binseg.configs.datasets.chasedb1.first_annotator import (
    dataset as _chase,
)
from bob.ip.binseg.configs.datasets.hrf.default import dataset as _hrf
from bob.ip.binseg.configs.datasets.iostar.vessel import dataset as _iostar

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
