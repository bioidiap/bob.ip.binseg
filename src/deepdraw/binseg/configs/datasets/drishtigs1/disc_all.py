# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""DRISHTI-GS1 dataset for Optic Disc Segmentation (agreed by all annotators)

* Configuration resolution: 1760 x 2048 (after center cropping)
* Reference (includes split): [DRISHTIGS1-2014]_
* See :py:mod:`deepdraw.binseg.data.drishtigs1` for dataset details
"""

from . import _maker

dataset = _maker("optic-disc-all")
