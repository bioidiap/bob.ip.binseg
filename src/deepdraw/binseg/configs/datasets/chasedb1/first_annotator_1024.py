# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""CHASE-DB1 dataset for Vessel Segmentation.

Configuration resolution: 1024 x 1024 (after Pad and resize)
"""

from . import _maker_square

dataset = _maker_square("first-annotator", 1024)
