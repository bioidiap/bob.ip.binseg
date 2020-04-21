#!/usr/bin/env python
# coding=utf-8

"""DRIVE dataset for Vessel Segmentation (default protocol)

* Split reference: [DRIVE-2004]_
* This configuration resolution: 544 x 544 (center-crop)
* See :py:mod:`bob.ip.binseg.data.drive` for dataset details
* This dataset offers a second-annotator comparison for the test set only
"""

from bob.ip.binseg.configs.datasets.drive import _maker
dataset = _maker("default")
second_annotator = _maker("second-annotator")
