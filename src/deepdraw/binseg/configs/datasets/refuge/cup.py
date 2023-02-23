#!/usr/bin/env python

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Tim Laibacher, tim.laibacher@idiap.ch
# SPDX-FileContributor: Oscar Jiménez del Toro, oscar.jimenez@idiap.ch
# SPDX-FileContributor: Maxime Délitroz, maxime.delitroz@idiap.ch
# SPDX-FileContributor: Andre Anjos andre.anjos@idiap.ch
# SPDX-FileContributor: Daniel Carron, daniel.carron@idiap.ch
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""REFUGE dataset for Optic Cup Segmentation (default protocol)

* Configuration resolution: 1632 x 1632 (after resizing and padding)
* Reference (including split): [REFUGE-2018]_
* See :py:mod:`deepdraw.binseg.data.refuge` for dataset details
"""

from deepdraw.binseg.configs.datasets.refuge import _maker

dataset = _maker("optic-cup")