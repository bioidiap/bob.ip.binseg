#!/usr/bin/env python
# coding=utf-8

"""CHASE-DB1 dataset for Vessel Segmentation (first-annotator protocol)

* Split reference: [CHASEDB1-2012]_
* Configuration resolution: 960 x 960 (after hand-specified crop)
* See :py:mod:`bob.ip.binseg.data.chasedb1` for dataset details
* This dataset offers a second-annotator comparison
"""

from bob.ip.binseg.configs.datasets.chasedb1 import _maker
dataset = _maker("first-annotator")
second_annotator = _maker("second-annotator")
