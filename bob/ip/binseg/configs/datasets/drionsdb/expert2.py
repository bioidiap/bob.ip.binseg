#!/usr/bin/env python
# coding=utf-8

"""DRIONS-DB for Optic Disc Segmentation (expert #2 annotations)

* Configuration resolution: 416 x 608 (after padding)
* Split reference: [MANINIS-2016]_
* See :py:mod:`bob.ip.binseg.data.drionsdb` for dataset details
"""

from bob.ip.binseg.configs.datasets.drionsdb import _maker
dataset = _maker("expert2")
