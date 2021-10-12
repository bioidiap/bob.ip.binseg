#!/usr/bin/env python
# coding=utf-8

"""JSRT CXR cross-evaluation dataset
"""

from bob.ip.binseg.configs.datasets.jsrt.default import dataset as _jsrt
from bob.ip.binseg.configs.datasets.montgomery.default import dataset as _mc
from bob.ip.binseg.configs.datasets.shenzhen.default import dataset as _shenzhen

dataset = {
    "train": _jsrt["train"],
    "validation": _jsrt["validation"],
    "test": _jsrt["test"],
    "montgomery (train)": _mc["train"],
    "montgomery (validation)": _mc["validation"],
    "montgomery (test)": _mc["test"],
    "shenzhen (train)": _shenzhen["train"],
    "shenzhen (validation)": _shenzhen["validation"],
    "shenzhen (test)": _shenzhen["test"],
}
