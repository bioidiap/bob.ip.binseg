# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""STARE dataset for Vessel Segmentation (annotator AH)

* Configuration resolution: 704 x 608 (after padding)
* Split reference: [MANINIS-2016]_
* See :py:mod:`deepdraw.data.stare` for dataset details
* This dataset offers a second-annotator comparison (using protocol "vk")
"""

from . import _maker

dataset = _maker("ah")
second_annotator = _maker("vk")
