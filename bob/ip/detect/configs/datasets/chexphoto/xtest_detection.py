#!/usr/bin/env python
# coding=utf-8

"""CheXphoto cross-evaluation dataset."""

from bob.ip.binseg.configs.datasets.chexphoto.default_detection import (
    dataset as _chexphoto,
)
from bob.ip.binseg.configs.datasets.jsrt.default_detection import (
    dataset as _jsrt,
)
from bob.ip.binseg.configs.datasets.montgomery.default_detection import (
    dataset as _mc,
)
from bob.ip.binseg.configs.datasets.shenzhen.default_detection import (
    dataset as _shenzhen,
)

dataset = {
    "train": _chexphoto["train"],
    "validation": _chexphoto["validation"],
    "test": _chexphoto["test"],
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
