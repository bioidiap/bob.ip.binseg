#!/usr/bin/env python
# coding=utf-8

"""IOSTAR dataset for Vessel Segmentation (default protocol)

* Split reference: [MEYER-2017]_
* Configuration resolution: 1024 x 1024 (original resolution)
* See :py:mod:`bob.ip.binseg.data.iostar` for dataset details
"""

from bob.ip.binseg.configs.datasets.iostar import _maker
dataset = _maker("vessel")
