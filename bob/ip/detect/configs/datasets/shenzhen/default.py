#!/usr/bin/env python
# coding=utf-8

"""Shenzhen dataset for Lung Detection (default protocol)

* Split reference: [GAAL-2020]_
* Configuration resolution: 256 x 256
* See :py:mod:`bob.ip.detect.data.shenzhen` for dataset details
"""

from bob.ip.detect.configs.datasets.shenzhen import _maker_augmented

dataset = _maker_augmented("default", 256)
