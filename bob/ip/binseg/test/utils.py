#!/usr/bin/env python
# coding=utf-8


"""Test utilities"""


import functools
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
