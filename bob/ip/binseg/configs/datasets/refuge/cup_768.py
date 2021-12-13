#!/usr/bin/env python
# coding=utf-8

"""REFUGE dataset for Optic Cup Segmentation

* Configuration resolution: 768 x 768 (after resizing and padding)

"""

from bob.ip.binseg.configs.datasets.refuge import _maker_square

dataset = _maker_square("optic-cup", 768)
