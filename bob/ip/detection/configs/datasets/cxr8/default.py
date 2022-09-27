#!/usr/bin/env python
# coding=utf-8

"""CXR8 Dataset (default protocol)

* Split reference: [GAAL-2020]_
* Configuration resolution: 256 x 256
* See :py:mod:`bob.ip.binseg.data.cxr8` for dataset details
"""

from bob.ip.binseg.configs.datasets.cxr8 import _maker_augmented

dataset = _maker_augmented("default", 256)
