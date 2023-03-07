# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Test utilities."""

import traceback

from typing import Optional

import click
import click.testing
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


def assert_click_runner_result(
    result: click.testing.Result,
    exit_code: int = 0,
    exception_type: Optional[type] = None,
):
    """Helper for asserting click runner results.

    Parameters
    ----------
    result
        The return value on ``click.testing.CLIRunner.invoke()``.
    exit_code
        The expected command exit code (defaults to 0).
    exception_type
        If given, will ensure that the raised exception is of that type.
    """

    m = (
        "Click command exited with code '{}', instead of '{}'.\n"
        "Exception:\n{}\n"
        "Output:\n{}"
    )
    exception = (
        "None"
        if result.exc_info is None
        else "".join(traceback.format_exception(*result.exc_info))
    )
    m = m.format(result.exit_code, exit_code, exception, result.output)
    assert result.exit_code == exit_code, m
    if exit_code == 0:
        assert not result.exception, m
    if exception_type is not None:
        assert isinstance(result.exception, exception_type), m
