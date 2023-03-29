# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shenzhen dataset for Lung Segmentation (default protocol)

* Split reference: [GAAL-2020]_
* Configuration resolution: 256 x 256 pixels
* See :py:mod:`deepdraw.data.shenzhen` for dataset details
"""

from deepdraw.configs.datasets.shenzhen import _maker

dataset = _maker("default", 256)
