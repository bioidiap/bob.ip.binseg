#!/usr/bin/env python
# coding=utf-8

"""DRISHTI-GS1 dataset for Optic Disc Segmentation (agreed by all annotators)

* Configuration resolution: 768 x 768 (after center cropping, padding and resizing)

"""

from bob.ip.binseg.configs.datasets.refuge import _maker_square

dataset = _maker_square("optic-disc", 768)
