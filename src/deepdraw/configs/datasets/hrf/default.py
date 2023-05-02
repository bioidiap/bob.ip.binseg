# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""HRF dataset for Vessel Segmentation (default protocol)

* Split reference: [ORLANDO-2017]_
* Configuration resolution: 1168 x 1648 (about half full HRF resolution)
* See :py:mod:`deepdraw.data.hrf` for dataset details
"""

from . import _maker_1168
from .default_fullres import dataset as _fr

dataset = _maker_1168("default")
dataset["train (full resolution)"] = _fr["train"]
dataset["test (full resolution)"] = _fr["test"]
