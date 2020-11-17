#!/usr/bin/env python
# coding=utf-8


"""Test utilities"""


import numpy


def count_bw(b):
    """Calculates totals of black and white pixels in a binary image


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
    return (boolean_array.size-white), white
