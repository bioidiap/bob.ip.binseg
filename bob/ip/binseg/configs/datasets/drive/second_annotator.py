#!/usr/bin/env python
# coding=utf-8

"""DRIVE dataset for Vessel Segmentation (second annotation: test only)

* Split reference: [DRIVE-2004]_
* This configuration resolution: 544 x 544 (center-crop)
* See :py:mod:`bob.ip.binseg.data.drive` for dataset details
* There are **NO training samples** on this configuration
"""

from bob.ip.binseg.configs.datasets.drive import _maker
dataset = _maker("second-annotator")
