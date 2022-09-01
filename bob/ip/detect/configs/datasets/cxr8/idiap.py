#!/usr/bin/env python
# coding=utf-8

"""CXR8 Dataset ("idiap" protocol - just like "default", but works at Idiap)

* Split reference: [GAAL-2020]_
* Configuration resolution: 256 x 256
* See :py:mod:`bob.ip.detect.data.cxr8` for dataset details
"""

from bob.ip.detect.configs.datasets.cxr8 import _maker_augmented

dataset = _maker_augmented("idiap", 256)