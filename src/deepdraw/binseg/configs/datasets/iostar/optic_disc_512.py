# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""IOSTAR dataset for Optic Disc Segmentation.

Configuration resolution: 512 x 512 (Resized )
"""

from . import _maker_square

dataset = _maker_square("optic-disc", 512)
