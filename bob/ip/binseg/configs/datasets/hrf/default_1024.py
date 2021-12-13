#!/usr/bin/env python
# coding=utf-8

"""HRF dataset for Vessel Segmentation

Configuration resolution: 1024 x 1024 (Pad + Resize)

"""

from bob.ip.binseg.configs.datasets.hrf import _maker_square

dataset = _maker_square("default", 1024)
