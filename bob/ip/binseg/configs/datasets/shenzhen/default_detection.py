#!/usr/bin/env python
# coding=utf-8

"""Shenzhen dataset for Lung Detection (default protocol)

* Split reference: [GAAL-2020]_
* Configuration resolution: 256 x 256
* See :py:mod:`bob.ip.binseg.data.shenzhen` for dataset details
"""

from bob.ip.binseg.configs.datasets.shenzhen import _maker_detection

dataset = _maker_detection("default", 256)
