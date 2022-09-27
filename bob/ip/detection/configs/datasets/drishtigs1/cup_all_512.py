#!/usr/bin/env python
# coding=utf-8

"""DRISHTI-GS1 dataset for Cup Segmentation (agreed by all annotators)

* Configuration resolution: 512 x 512 (after center cropping, padding and resizing)

"""

from bob.ip.binseg.configs.datasets.drishtigs1 import _maker_square

dataset = _maker_square("optic-cup-all", 512)
