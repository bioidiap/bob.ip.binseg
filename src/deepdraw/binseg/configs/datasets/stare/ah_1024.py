# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""STARE dataset for Vessel Segmentation (annotator AH)

Configuration resolution: 1024 x 1024 (after padding and resizing)
"""

from . import _maker_square

dataset = _maker_square("ah", 1024)
