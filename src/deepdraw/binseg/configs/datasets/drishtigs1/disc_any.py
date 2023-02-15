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

"""DRISHTI-GS1 dataset for Optic Disc Segmentation (agreed by any annotator)

* Configuration resolution: 1760 x 2048 (after center cropping)
* Reference (includes split): [DRISHTIGS1-2014]_
* See :py:mod:`bob.ip.binseg.data.drishtigs1` for dataset details
"""

from bob.ip.binseg.configs.datasets.drishtigs1 import _maker

dataset = _maker("optic-disc-any")
