#!/usr/bin/env python
# coding=utf-8

"""HRF dataset for Vessel Segmentation

Configuration resolution: 1024 x 1024 (Pad + Resize)

"""

from bob.ip.binseg.configs.datasets.hrf import _maker_square_1024

dataset = _maker_square_1024("default")
