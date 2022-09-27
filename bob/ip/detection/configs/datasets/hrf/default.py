#!/usr/bin/env python
# coding=utf-8

"""HRF dataset for Vessel Segmentation (default protocol)

* Split reference: [ORLANDO-2017]_
* Configuration resolution: 1168 x 1648 (about half full HRF resolution)
* See :py:mod:`bob.ip.binseg.data.hrf` for dataset details
"""

from bob.ip.binseg.configs.datasets.hrf import _maker_1168

dataset = _maker_1168("default")

from bob.ip.binseg.configs.datasets.hrf.default_fullres import dataset as _fr

dataset["train (full resolution)"] = _fr["train"]
dataset["test (full resolution)"] = _fr["test"]
