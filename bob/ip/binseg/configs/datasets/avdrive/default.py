#!/usr/bin/env python
# coding=utf-8

"""DRIVE dataset for Vessel Segmentation (default protocol)

* Split reference: [DRIVE-2004]_
* This configuration resolution: 544 x 544 (center-crop)
* See :py:mod:`bob.ip.binseg.data.drive` for dataset details
* We are using DRIVE dataset for artery vein segmentation
"""

from bob.ip.binseg.configs.datasets.avdrive import _maker

dataset = _maker("default")
