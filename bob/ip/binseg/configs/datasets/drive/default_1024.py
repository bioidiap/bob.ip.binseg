#!/usr/bin/env python
# coding=utf-8

"""DRIVE dataset for Vessel Segmentation (Resolution used for MTL models)

This configuration resolution: 1024 x 1024 (Pad + resize)

"""

from bob.ip.binseg.configs.datasets.drive import _maker_square

dataset = _maker_square("default", 1024)
