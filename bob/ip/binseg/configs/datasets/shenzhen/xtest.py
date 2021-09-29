#!/usr/bin/env python
# coding=utf-8

"""Shenzhen cross-evaluation dataset
"""

from bob.ip.binseg.configs.datasets.jsrt.default import dataset as _jsrt
from bob.ip.binseg.configs.datasets.montgomery.default import dataset as _mc
from bob.ip.binseg.configs.datasets.shenzhen.default import dataset as _shenzhen

dataset = {
    "train": _shenzhen["train"],
    "validation": _shenzhen["validation"],
    "test": _shenzhen["test"],
    "montgomery (train)": _mc["train"],
    "montgomery (validation)": _mc["validation"],
    "montgomery (test)": _mc["test"],
    "jsrt (train)": _jsrt["train"],
    "jsrt (validation)": _jsrt["validation"],
    "jsrt (test)": _jsrt["test"],
}
