#!/usr/bin/env python
# coding=utf-8

"""Montgomery County dataset for Lung Segmentation (default protocol)

* Split reference: [GAAL-2020]_
* Configuration resolution: 256 x 256
* See :py:mod:`bob.ip.detect.data.montgomery` for dataset details
"""

from bob.ip.detect.configs.datasets.montgomery import _maker_augmented

dataset = _maker_augmented("default")