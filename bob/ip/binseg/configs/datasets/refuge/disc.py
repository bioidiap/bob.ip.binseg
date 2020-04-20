#!/usr/bin/env python
# coding=utf-8

"""REFUGE dataset for Optic Disc Segmentation (default protocol)

* Configuration resolution: 1632 x 1632 (after resizing and padding)
* Reference (including split): [REFUGE-2018]_
* See :py:mod:`bob.ip.binseg.data.refuge` for dataset details
"""

from bob.ip.binseg.configs.datasets.refuge import _maker
dataset = _maker("optic-disc")
