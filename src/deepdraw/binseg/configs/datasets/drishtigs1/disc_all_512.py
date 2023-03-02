# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""DRISHTI-GS1 dataset for Optic Disc Segmentation (agreed by all annotators)

* Configuration resolution: 512 x 512 (after center cropping, padding and resizing)
"""

from . import _maker_square

dataset = _maker_square("optic-disc-all", 512)
