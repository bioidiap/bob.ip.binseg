#!/usr/bin/env python
# coding=utf-8

"""JSRT cross-evaluation dataset
"""

from bob.ip.binseg.configs.datasets.JSRT.default import dataset as _jsrt
from bob.ip.binseg.configs.datasets.MC.default import dataset as _mc
from bob.ip.binseg.configs.datasets.Shenzhen.default import dataset as _shenzhen

dataset = {
    "train": _jsrt["train"],
    "validation": _jsrt["validation"],
    "test": _jsrt["test"],
    "MC (train)": _mc["train"],
    "MC (validation)": _mc["validation"],
    "MC (test)": _mc["test"],
    "Shenzhen (train)": _shenzhen["train"],
    "Shenzhen (validation)": _shenzhen["validation"],
    "Shenzhen (test)": _shenzhen["test"],
}
