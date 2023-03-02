# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""REFUGE dataset for Optic Cup Segmentation.

* Configuration resolution: 768 x 768 (after resizing and padding)
"""

from . import _maker_square

dataset = _maker_square("optic-cup", 768)
