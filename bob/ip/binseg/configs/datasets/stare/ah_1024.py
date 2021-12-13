#!/usr/bin/env python
# coding=utf-8

"""STARE dataset for Vessel Segmentation (annotator AH)

Configuration resolution: 1024 x 1024 (after padding and resizing)

"""
from bob.ip.binseg.configs.datasets.stare import _maker_square

dataset = _maker_square("ah", 1024)
