#!/usr/bin/env python
# coding=utf-8

"""DRHAGIS dataset for Vessel Segmentation (default protocol)


* This configuration resolution: 1880 x 1880 (center-crop)

"""

from bob.ip.binseg.configs.datasets.drhagis import _maker
dataset = _maker("default")

