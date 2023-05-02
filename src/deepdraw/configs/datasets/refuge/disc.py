# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""REFUGE dataset for Optic Disc Segmentation (default protocol)

* Configuration resolution: 1632 x 1632 (after resizing and padding)
* Reference (including split): [REFUGE-2018]_
* See :py:mod:`deepdraw.data.refuge` for dataset details
"""

from . import _maker

dataset = _maker("optic-disc")
