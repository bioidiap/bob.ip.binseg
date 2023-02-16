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

"""DRIONS-DB for Optic Disc Segmentation (expert #1 annotations)

* Configuration resolution: 416 x 608 (after padding)
* Split reference: [MANINIS-2016]_
* See :py:mod:`deepdraw.binseg.data.drionsdb` for dataset details
"""

from deepdraw.binseg.configs.datasets.drionsdb import _maker

dataset = _maker("expert1")
