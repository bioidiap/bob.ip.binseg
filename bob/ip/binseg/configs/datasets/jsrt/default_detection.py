#!/usr/bin/env python
# coding=utf-8

"""Japanese Society of Radiological Technology dataset for Detection (default protocol)

* Split reference: [GAAL-2020]_
* Configuration resolution: 256 x 256
* See :py:mod:`bob.ip.binseg.data.jsrt` for dataset details
"""

from bob.ip.binseg.configs.datasets.jsrt import _maker_detection

dataset = _maker_detection("default")
