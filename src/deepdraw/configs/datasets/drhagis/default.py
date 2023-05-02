# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""DRHAGIS dataset for Vessel Segmentation (default protocol)

* This configuration resolution: 1760 x 1760 (Resizing)
"""

from . import _maker

dataset = _maker("default")
