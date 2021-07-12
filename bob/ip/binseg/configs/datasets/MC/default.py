#!/usr/bin/env python
# coding=utf-8

"""Montgomery County dataset for Lung Segmentation (default protocol)

* Split reference: [GAÁL-2020]_
* Configuration resolution: 512 x 512
* See :py:mod:`bob.ip.binseg.data.MC` for dataset details
"""

from bob.ip.binseg.configs.datasets.MC import _maker

dataset = _maker("default")
