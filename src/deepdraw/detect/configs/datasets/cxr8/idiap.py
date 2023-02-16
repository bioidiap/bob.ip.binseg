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

"""CXR8 Dataset ("idiap" protocol - just like "default", but works at Idiap)

* Split reference: [GAAL-2020]_
* Configuration resolution: 256 x 256
* See :py:mod:`deepdraw.detect.data.cxr8` for dataset details
"""

from deepdraw.detect.configs.datasets.cxr8 import _maker_augmented

dataset = _maker_augmented("idiap", 256)
