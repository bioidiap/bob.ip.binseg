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

"""HRF dataset for Vessel Segmentation (default protocol)

* Split reference: [ORLANDO-2017]_
* Configuration resolution: 2336 x 3296 (full dataset resolution)
* See :py:mod:`deepdraw.binseg.data.hrf` for dataset details
"""

from deepdraw.binseg.configs.datasets.hrf import _maker

dataset = _maker("default")
