#!/usr/bin/env python
# coding=utf-8

"""IOSTAR dataset for Optic Disc Segmentation

Configuration resolution: 768 x 768 (Resized )

"""

from bob.ip.binseg.configs.datasets.iostar import _maker_square_768

dataset = _maker_square_768("optic-disc")
