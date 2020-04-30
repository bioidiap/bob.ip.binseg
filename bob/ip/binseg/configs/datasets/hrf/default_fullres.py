#!/usr/bin/env python
# coding=utf-8

"""HRF dataset for Vessel Segmentation (default protocol)

* Split reference: [ORLANDO-2017]_
* Configuration resolution: 2336 x 3296 (full dataset resolution)
* See :py:mod:`bob.ip.binseg.data.hrf` for dataset details
"""

from bob.ip.binseg.configs.datasets.hrf import _maker
dataset = _maker("default")
