#!/usr/bin/env python
# coding=utf-8

"""RIM-ONE r3 for Optic Disc Segmentation (expert #2 annotations)

* Configuration resolution: 1440 x 1088 (after padding)
* Split reference: [MANINIS-2016]_
* See :py:mod:`bob.ip.binseg.data.rimoner3` for dataset details
"""

from bob.ip.binseg.configs.datasets.rimoner3 import _maker
dataset = _maker("optic-disc-exp2")
