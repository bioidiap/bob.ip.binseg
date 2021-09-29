#!/usr/bin/env python
# coding=utf-8

"""Montgomery County dataset for Lung Segmentation (default protocol)

* Split reference: [GAAL-2020]_
* Configuration resolution: 512 x 512
* See :py:mod:`bob.ip.binseg.data.montgomery` for dataset details
"""

from bob.ip.binseg.configs.datasets.montgomery import _maker

dataset = _maker("default")
