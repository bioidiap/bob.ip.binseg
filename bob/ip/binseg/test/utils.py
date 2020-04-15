#!/usr/bin/env python
# coding=utf-8


"""Test utilities"""


import functools

import numpy
import nose.plugins.skip

import bob.extension


def rc_variable_set(name):
    """
    Decorator that checks if a given bobrc variable is set before running
    """

    def wrapped_function(test):
        @functools.wraps(test)
        def wrapper(*args, **kwargs):
            if name not in bob.extension.rc:
                raise nose.plugins.skip.SkipTest("Bob's RC variable '%s' is not set" % name)
            return test(*args, **kwargs)

        return wrapper

    return wrapped_function


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
