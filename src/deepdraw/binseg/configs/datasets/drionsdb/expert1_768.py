# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""DRIONS-DB for Optic Disc Segmentation (expert #1 annotations)

Configuration resolution: 768x768 (after padding and resizing)
"""

from . import _maker_square

dataset = _maker_square("expert1", 768)
