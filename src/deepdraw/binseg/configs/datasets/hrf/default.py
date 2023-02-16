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
* Configuration resolution: 1168 x 1648 (about half full HRF resolution)
* See :py:mod:`deepdraw.binseg.data.hrf` for dataset details
"""

from deepdraw.binseg.configs.datasets.hrf import _maker_1168

dataset = _maker_1168("default")

from deepdraw.binseg.configs.datasets.hrf.default_fullres import dataset as _fr

dataset["train (full resolution)"] = _fr["train"]
dataset["test (full resolution)"] = _fr["test"]
