#!/usr/bin/env python
# coding=utf-8

"""Montgomery County cross-evaluation dataset
"""

from bob.ip.binseg.configs.datasets.jsrt.default import dataset as _jsrt
from bob.ip.binseg.configs.datasets.montgomery.default import dataset as _mc
from bob.ip.binseg.configs.datasets.shenzhen.default import dataset as _shenzhen

dataset = {
    "train": _mc["train"],
    "validation": _mc["validation"],
    "test": _mc["test"],
    "jsrt (train)": _jsrt["train"],
    "jsrt (validation)": _jsrt["validation"],
    "jsrt (test)": _jsrt["test"],
    "shenzhen (train)": _shenzhen["train"],
    "shenzhen (validation)": _shenzhen["validation"],
    "shenzhen (test)": _shenzhen["test"],
}
