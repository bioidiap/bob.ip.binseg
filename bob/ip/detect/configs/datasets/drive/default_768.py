#!/usr/bin/env python
# coding=utf-8

"""DRIVE dataset for Vessel Segmentation (Resolution used for MTL models)

This configuration resolution: 768 x 768 (Pad + resize)

"""

from bob.ip.binseg.configs.datasets.drive import _maker_square

dataset = _maker_square("default", 768)
