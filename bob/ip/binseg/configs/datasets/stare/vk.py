#!/usr/bin/env python
# coding=utf-8

"""STARE dataset for Vessel Segmentation (annotator VK)

* Configuration resolution: 704 x 608 (after padding)
* Split reference: [MANINIS-2016]_
* See :py:mod:`bob.ip.binseg.data.stare` for dataset details
* This dataset offers a second-annotator comparison (using protocol "ah")
"""

from bob.ip.binseg.configs.datasets.stare import _maker
dataset = _maker("vk")
second_annotator = _maker("ah")
