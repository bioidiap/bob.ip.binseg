#!/usr/bin/env python
# coding=utf-8

"""Shenzhen cross-evaluation dataset
"""

from bob.ip.binseg.configs.datasets.Shenzhen.default import (
    dataset as _shenzhen
)
from bob.ip.binseg.configs.datasets.JSRT.default import dataset as _jsrt
from bob.ip.binseg.configs.datasets.MC.default import (
    dataset as _mc,
)


dataset = {
    "train": _shenzhen["train"],
    "validation": _shenzhen["validation"],
    "test": _shenzhen["test"],
    "MC (train)": _mc["train"],
    "MC (validation)": _mc["validation"],
    "MC (test)": _mc["test"],
    "JSRT (train)": _jsrt["train"],
    "JSRT (validation)": _jsrt["validation"],
    "JSRT (test)": _jsrt["test"]
}
