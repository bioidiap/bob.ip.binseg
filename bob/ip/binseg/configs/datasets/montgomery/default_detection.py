#!/usr/bin/env python
# coding=utf-8

"""Montgomery County dataset for Lung Detection (default protocol)

* Split reference: [GAAL-2020]_
* Configuration resolution: 256 x 256
* See :py:mod:`bob.ip.binseg.data.montgomery` for dataset details
"""

from bob.ip.binseg.configs.datasets.montgomery import _maker_detection

dataset = _maker_detection("default")
