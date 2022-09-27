#!/usr/bin/env python
# coding=utf-8

"""DRIONS-DB for Optic Disc Segmentation (expert #2 annotations)

Configuration resolution: 512x512 (after padding and resizing)

"""

from bob.ip.binseg.configs.datasets.drionsdb import _maker_square

dataset = _maker_square("expert2", 512)
