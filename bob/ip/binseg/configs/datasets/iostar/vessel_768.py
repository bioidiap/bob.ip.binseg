#!/usr/bin/env python
# coding=utf-8

"""IOSTAR dataset for Vessel Segmentation (default protocol)

Configuration resolution: 768 x 768 (Resize)

"""

from bob.ip.binseg.configs.datasets.iostar import _maker_vessel_square

dataset = _maker_vessel_square("vessel")
