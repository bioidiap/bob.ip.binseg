# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""DRIVE dataset for Vessel Segmentation (default protocol)

* Split reference: [DRIVE-2004]_
* This configuration resolution: 544 x 544 (center-crop)
* See :py:mod:`deepdraw.binseg.data.drive` for dataset details
* We are using DRIVE dataset for artery vein segmentation
"""

from . import _maker

dataset = _maker("default")
