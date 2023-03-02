# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shenzhen dataset for Lung Detection (default protocol)

* Split reference: [GAAL-2020]_
* Configuration resolution: 256 x 256
* See :py:mod:`deepdraw.detect.data.shenzhen` for dataset details
"""

from deepdraw.detect.configs.datasets.shenzhen import _maker_augmented

dataset = _maker_augmented("default", 256)
