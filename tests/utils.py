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

"""Test utilities."""


import numpy


def count_bw(b):
    """Calculates totals of black and white pixels in a binary image.

    Parameters
    ----------

    b : PIL.Image.Image
        A PIL image in mode "1" to be used for calculating positives and
        negatives

    Returns
    -------

    black : int
        Number of black pixels in the binary image

    white : int
        Number of white pixels in the binary image
    """

    boolean_array = numpy.array(b)
    white = boolean_array.sum()
    return (boolean_array.size - white), white
