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

"""CHASE-DB1 dataset for Vessel Segmentation (first-annotator protocol)

* Split reference: [CHASEDB1-2012]_
* Configuration resolution: 960 x 960 (after hand-specified crop)
* See :py:mod:`deepdraw.binseg.data.chasedb1` for dataset details
* This dataset offers a second-annotator comparison
"""

from deepdraw.binseg.configs.datasets.chasedb1 import _maker

dataset = _maker("first-annotator")
second_annotator = _maker("second-annotator")
