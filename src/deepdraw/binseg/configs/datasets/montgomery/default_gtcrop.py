# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Montgomery County dataset for Lung Segmentation (default protocol)

* Split reference: [GAAL-2020]_
* Configuration resolution: 256 x 256
* See :py:mod:`deepdraw.binseg.data.montgomery` for dataset details
"""

from . import _maker_augmented_gt_box

dataset = _maker_augmented_gt_box("default")
