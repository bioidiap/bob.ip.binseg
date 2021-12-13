#!/usr/bin/env python
# coding=utf-8

"""RIM-ONE r3 for Optic Cup Segmentation (expert #1 annotations)

Configuration resolution: 512 x 512 (after padding and resizing)

"""

from bob.ip.binseg.configs.datasets.rimoner3 import _maker_square_512

dataset = _maker_square_512("optic-cup-exp1")
