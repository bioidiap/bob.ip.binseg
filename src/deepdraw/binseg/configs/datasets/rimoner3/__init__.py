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


def _maker(protocol):
    from .....common.data.transforms import Pad
    from ....data.rimoner3 import dataset as raw
    from .. import make_dataset as mk

    return mk(raw.subsets(protocol), [Pad((8, 8, 8, 8))])


def _maker_square(protocol, size):
    from .....common.data.transforms import Pad, Resize
    from ....data.rimoner3 import dataset as raw
    from .. import make_dataset as mk

    return mk(raw.subsets(protocol), [Pad((176, 0)), Resize((size, size))])
