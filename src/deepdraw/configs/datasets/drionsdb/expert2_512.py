# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""DRIONS-DB for Optic Disc Segmentation (expert #2 annotations)

Configuration resolution: 512x512 (after padding and resizing)
"""

from . import _maker_square

dataset = _maker_square("expert2", 512)
