#!/usr/bin/env python
# coding=utf-8

"""DRIONS-DB for Optic Disc Segmentation (expert #2 annotations)

Configuration resolution: 768x768 (after padding and resizing)

"""

from bob.ip.binseg.configs.datasets.drionsdb import _maker_square

dataset = _maker_square("expert2", 768)
