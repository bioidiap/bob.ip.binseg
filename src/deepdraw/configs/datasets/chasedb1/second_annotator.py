# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""CHASE-DB1 dataset for Vessel Segmentation (second-annotator protocol)

* Split reference: [CHASEDB1-2012]_
* Configuration resolution: 960 x 960 (after hand-specified crop)
* See :py:mod:`deepdraw.data.chasedb1` for dataset details
* This dataset offers a second-annotator comparison (using "first-annotator")
"""

from . import _maker

dataset = _maker("second-annotator")
second_annotator = _maker("first-annotator")
