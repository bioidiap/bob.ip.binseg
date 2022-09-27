#!/usr/bin/env python
# coding=utf-8

"""CXR8 cross-evaluation dataset with Idiap directory structure organisation
"""

from bob.ip.detect.configs.datasets.cxr8.idiap import dataset as _cxr8
from bob.ip.detect.configs.datasets.jsrt.default import dataset as _jsrt
from bob.ip.detect.configs.datasets.montgomery.default import dataset as _mc
from bob.ip.detect.configs.datasets.shenzhen.default import dataset as _shenzhen

dataset = {
    "train": _cxr8["train"],
    "validation": _cxr8["validation"],
    "test": _cxr8["test"],
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
