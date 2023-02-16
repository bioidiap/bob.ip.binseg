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

"""Japanese Society of Radiological Technology dataset for Lung Segmentation
(default protocol)

* Split reference: [GAAL-2020]_
* Configuration resolution: 256 x 256
* See :py:mod:`deepdraw.binseg.data.jsrt` for dataset details
"""

from deepdraw.binseg.configs.datasets.jsrt import _maker_augmented

dataset = _maker_augmented("default")
