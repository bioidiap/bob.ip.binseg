# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""DRIONS-DB for Optic Disc Segmentation (expert #2 annotations)

* Configuration resolution: 416 x 608 (after padding)
* Split reference: [MANINIS-2016]_
* See :py:mod:`deepdraw.data.drionsdb` for dataset details
"""

from . import _maker

dataset = _maker("expert2")
