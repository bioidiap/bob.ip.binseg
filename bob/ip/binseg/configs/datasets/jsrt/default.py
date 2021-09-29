#!/usr/bin/env python
# coding=utf-8

"""Japanese Society of Radiological Technology dataset for Lung Segmentation (default protocol)

* Split reference: [GAAL-2020]_
* Configuration resolution: 512 x 512
* See :py:mod:`bob.ip.binseg.data.jsrt` for dataset details
"""

from bob.ip.binseg.configs.datasets.jsrt import _maker

dataset = _maker("default")
