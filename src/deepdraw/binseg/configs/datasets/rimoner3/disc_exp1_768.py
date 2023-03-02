# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""RIM-ONE r3 for Optic Disc Segmentation (expert #1 annotations)

Configuration resolution: 768 x 768 (after padding and resizing)
"""

from . import _maker_square

dataset = _maker_square("optic-disc-exp1", 768)
