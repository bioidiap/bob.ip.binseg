#!/usr/bin/env python
# coding=utf-8

"""CheXphoto dataset for Lung Detection (default protocol).

* Split reference: [GAAL-2020]_
* Configuration resolution: 256 x 256
* See :py:mod:`bob.ip.binseg.data.chexphoto` for dataset details
"""

from bob.ip.binseg.configs.datasets.chexphoto import _maker_augmented

dataset = _maker_augmented("default", detection=True)
