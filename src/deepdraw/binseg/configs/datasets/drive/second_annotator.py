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

"""DRIVE dataset for Vessel Segmentation (second annotation: test only)

* Split reference: [DRIVE-2004]_
* This configuration resolution: 544 x 544 (center-crop)
* See :py:mod:`deepdraw.binseg.data.drive` for dataset details
* There are **NO training samples** on this configuration
"""

from deepdraw.binseg.configs.datasets.drive import _maker

dataset = _maker("second-annotator")
