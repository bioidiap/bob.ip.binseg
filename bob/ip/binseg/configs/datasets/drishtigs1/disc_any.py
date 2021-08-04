#!/usr/bin/env python
# coding=utf-8

"""DRISHTI-GS1 dataset for Optic Disc Segmentation (agreed by any annotator)

* Configuration resolution: 1760 x 2048 (after center cropping)
* Reference (includes split): [DRISHTIGS1-2014]_
* See :py:mod:`bob.ip.binseg.data.drishtigs1` for dataset details
"""

from bob.ip.binseg.configs.datasets.drishtigs1 import _maker

dataset = _maker("optic-disc-any")
