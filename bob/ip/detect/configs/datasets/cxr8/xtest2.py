#!/usr/bin/env python
# coding=utf-8

"""CXR8 cross-evaluation dataset
"""

from bob.ip.binseg.configs.datasets.cxr8.default import dataset as _cxr8
from bob.ip.binseg.configs.datasets.jsrt.default_cxr8 import dataset as _jsrt
from bob.ip.binseg.configs.datasets.montgomery.default_cxr8 import dataset as _mc
from bob.ip.binseg.configs.datasets.shenzhen.default_cxr8 import dataset as _shenzhen

dataset = {
    "validation": _cxr8["validation"],
    "montgomery (train)": _mc["train"],
    "montgomery (validation)": _mc["validation"],
    "montgomery (test)": _mc["test"],
    "jsrt (train)": _jsrt["train"],
    "jsrt (validation)": _jsrt["validation"],
    "jsrt (test)": _jsrt["test"],
    "shenzhen (train)": _shenzhen["train"],
    "shenzhen (validation)": _shenzhen["validation"],
    "shenzhen (test)": _shenzhen["test"],
}
