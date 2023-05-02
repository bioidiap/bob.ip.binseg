# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""HRF dataset for Vessel Segmentation (default protocol)

* Split reference: [ORLANDO-2017]_
* Configuration resolution: 2336 x 3296 (full dataset resolution)
* See :py:mod:`deepdraw.data.hrf` for dataset details
"""

from . import _maker

dataset = _maker("default")
