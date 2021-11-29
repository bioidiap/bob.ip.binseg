#!/usr/bin/env python
# coding=utf-8

"""STARE dataset for Vessel Segmentation (annotator AH)

Configuration resolution: 768 x 768 (after padding and resizing)

"""
from bob.ip.binseg.configs.datasets.stare import _maker_square_768

dataset = _maker_square_768("ah")
