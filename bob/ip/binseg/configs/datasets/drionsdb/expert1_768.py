#!/usr/bin/env python
# coding=utf-8

"""DRIONS-DB for Optic Disc Segmentation (expert #1 annotations)

Configuration resolution: 768x768 (after padding and resizing)

"""

from bob.ip.binseg.configs.datasets.drionsdb import _maker_square

dataset = _maker_square("expert1", 768)
