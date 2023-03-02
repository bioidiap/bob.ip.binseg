# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""HRF dataset for Vessel Segmentation.

Configuration resolution: 768 x 768 (Pad + Resize)
"""

from . import _maker_square

dataset = _maker_square("default", 768)
