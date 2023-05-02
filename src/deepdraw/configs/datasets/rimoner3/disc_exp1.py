# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""RIM-ONE r3 for Optic Disc Segmentation (expert #1 annotations)

* Configuration resolution: 1440 x 1088 (after padding)
* Split reference: [MANINIS-2016]_
* See :py:mod:`deepdraw.data.rimoner3` for dataset details
"""

from . import _maker

dataset = _maker("optic-disc-exp1")
