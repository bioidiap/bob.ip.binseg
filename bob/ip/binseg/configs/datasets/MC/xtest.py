#!/usr/bin/env python
# coding=utf-8

"""MC cross-evaluation dataset
"""

from bob.ip.binseg.configs.datasets.JSRT.default import dataset as _jsrt
from bob.ip.binseg.configs.datasets.MC.default import dataset as _mc
from bob.ip.binseg.configs.datasets.Shenzhen.default import dataset as _shenzhen

dataset = {
    "train": _mc["train"],
    "validation": _mc["validation"],
    "test": _mc["test"],
    "JSRT (train)": _jsrt["train"],
    "JSRT (validation)": _jsrt["validation"],
    "JSRT (test)": _jsrt["test"],
    "Shenzhen (train)": _shenzhen["train"],
    "Shenzhen (validation)": _shenzhen["validation"],
    "Shenzhen (test)": _shenzhen["test"],
}
