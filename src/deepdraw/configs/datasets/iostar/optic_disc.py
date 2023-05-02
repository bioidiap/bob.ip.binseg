# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""IOSTAR dataset for Optic Disc Segmentation (default protocol)

* Split reference: [MEYER-2017]_
* Configuration resolution: 1024 x 1024 (original resolution)
* See :py:mod:`deepdraw.data.iostar` for dataset details
"""

from . import _maker

dataset = _maker("optic-disc")
