# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""IOSTAR dataset for Vessel Segmentation (default protocol)

Configuration resolution: 768 x 768 (Resize)
"""

from . import _maker_square

dataset = _maker_square("vessel", 768)
