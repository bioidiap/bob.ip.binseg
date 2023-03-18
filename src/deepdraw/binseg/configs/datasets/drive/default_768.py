# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""DRIVE dataset for Vessel Segmentation (Resolution used for MTL models)

This configuration resolution: 768 x 768 (Pad + resize)
"""

from . import _maker_square

dataset = _maker_square("default", 768)
