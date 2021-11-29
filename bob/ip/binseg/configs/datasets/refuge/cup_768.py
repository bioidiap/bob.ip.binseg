#!/usr/bin/env python
# coding=utf-8

"""REFUGE dataset for Optic Cup Segmentation

* Configuration resolution: 768 x 768 (after resizing and padding)

"""

from bob.ip.binseg.configs.datasets.refuge import _maker_square_768

dataset = _maker_square_768("optic-cup")
