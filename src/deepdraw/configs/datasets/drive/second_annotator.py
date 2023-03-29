# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""DRIVE dataset for Vessel Segmentation (second annotation: test only)

* Split reference: [DRIVE-2004]_
* This configuration resolution: 544 x 544 (center-crop)
* See :py:mod:`deepdraw.data.drive` for dataset details
* There are **NO training samples** on this configuration
"""

from . import _maker

dataset = _maker("second-annotator")
