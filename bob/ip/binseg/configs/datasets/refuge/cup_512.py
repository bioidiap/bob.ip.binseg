#!/usr/bin/env python
# coding=utf-8

"""REFUGE dataset for Optic Cup Segmentation

* Configuration resolution: 512 x 512 (after resizing and padding)

"""

from bob.ip.binseg.configs.datasets.refuge import _maker_square

dataset = _maker_square("optic-cup")
