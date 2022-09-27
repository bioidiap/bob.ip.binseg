#!/usr/bin/env python
# coding=utf-8

"""CHASE-DB1 dataset for Vessel Segmentation

Configuration resolution: 768 x 768 (after Pad and resize)

"""

from bob.ip.binseg.configs.datasets.chasedb1 import _maker_square

dataset = _maker_square("first-annotator", 768)
