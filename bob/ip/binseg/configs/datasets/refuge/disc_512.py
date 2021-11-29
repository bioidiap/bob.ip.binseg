#!/usr/bin/env python
# coding=utf-8

"""DRISHTI-GS1 dataset for Optic Disc Segmentation (agreed by all annotators)

* Configuration resolution: 512 x 512 (after center cropping, padding and resizing)

"""

from bob.ip.binseg.configs.datasets.drishtigs1 import _maker_square_512

dataset = _maker_square_512("optic-disc-all")
