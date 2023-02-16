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

"""IOSTAR dataset for Optic Disc Segmentation (default protocol)

* Split reference: [MEYER-2017]_
* Configuration resolution: 1024 x 1024 (original resolution)
* See :py:mod:`deepdraw.binseg.data.iostar` for dataset details
"""

from deepdraw.binseg.configs.datasets.iostar import _maker

dataset = _maker("optic-disc")
