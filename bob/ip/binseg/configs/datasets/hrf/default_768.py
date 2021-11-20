#!/usr/bin/env python
# coding=utf-8

"""HRF dataset for Vessel Segmentation

Configuration resolution: 768 x 768 (Pad + Resize)

"""

from bob.ip.binseg.configs.datasets.hrf import _maker_square

dataset = _maker_square("default")
