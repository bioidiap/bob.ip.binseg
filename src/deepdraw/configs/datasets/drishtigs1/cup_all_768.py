# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""DRISHTI-GS1 dataset for Cup Segmentation (agreed by all annotators)

* Configuration resolution: 768 x 768 (after center cropping, padding and resizing)
"""

from . import _maker_square

dataset = _maker_square("optic-cup-all", 768)
