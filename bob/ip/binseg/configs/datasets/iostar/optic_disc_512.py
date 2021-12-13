#!/usr/bin/env python
# coding=utf-8

"""IOSTAR dataset for Optic Disc Segmentation

Configuration resolution: 512 x 512 (Resized )

"""

from bob.ip.binseg.configs.datasets.iostar import _maker_square

dataset = _maker_square("optic-disc", 512)
