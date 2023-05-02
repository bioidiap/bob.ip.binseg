# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""CHASE-DB1 dataset for Vessel Segmentation (first-annotator protocol)

* Split reference: [CHASEDB1-2012]_
* Configuration resolution: 960 x 960 (after hand-specified crop)
* See :py:mod:`deepdraw.data.chasedb1` for dataset details
* This dataset offers a second-annotator comparison
"""

from . import _maker

dataset = _maker("first-annotator")
second_annotator = _maker("second-annotator")
