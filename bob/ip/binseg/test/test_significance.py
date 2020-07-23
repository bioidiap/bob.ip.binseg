#!/usr/bin/env python
# coding=utf-8

"""Tests for significance tools"""


import numpy
import pandas
import nose.tools
import torch

from ..engine.significance import (
    _winperf_measures,
    _performance_summary,
    PERFORMANCE_FIGURES,
)
from ..utils.measure import base_measures


def _check_window_measures(pred, gt, threshold, size, stride, expected):

    pred = torch.tensor(pred)
    gt = torch.tensor(gt)
    actual = _winperf_measures(pred, gt, threshold, size, stride)

    # transforms tp,tn,fp,fn through base_measures()
    expected_shape = numpy.array(expected).shape[:2]
    expected = numpy.array([base_measures(*c) for r in expected for c in r]).T
    expected = expected.reshape((len(PERFORMANCE_FIGURES),) + expected_shape)

    assert numpy.allclose(
        actual, expected
    ), f"Actual output:\n{actual}\n **!=** Expected output:\n{expected}"


def test_winperf_measures_alltrue():

    pred = numpy.ones((4, 4), dtype=float)
    gt = numpy.ones((4, 4), dtype=bool)
    threshold = 0.5
    size = (2, 2)
    stride = (1, 1)

    expected = [
        # tp, fp, tn, fn
        [(4, 0, 0, 0), (4, 0, 0, 0), (4, 0, 0, 0)],
        [(4, 0, 0, 0), (4, 0, 0, 0), (4, 0, 0, 0)],
        [(4, 0, 0, 0), (4, 0, 0, 0), (4, 0, 0, 0)],
    ]
    _check_window_measures(pred, gt, threshold, size, stride, expected)


def test_winperf_measures_alltrue_with_padding():

    pred = numpy.ones((3, 3), dtype=float)
    gt = numpy.ones((3, 3), dtype=bool)
    threshold = 0.5
    size = (2, 2)
    stride = (2, 2)

    expected = [
        # tp, fp, tn, fn
        [(4, 0, 0, 0), (2, 0, 2, 0)],
        [(2, 0, 2, 0), (1, 0, 3, 0)],
    ]
    _check_window_measures(pred, gt, threshold, size, stride, expected)


def test_winperf_measures_dot_with_padding():

    pred = numpy.ones((3, 3), dtype=float)
    gt = numpy.zeros((3, 3), dtype=bool)
    gt[1, 1] = 1.0  # white dot pattern
    threshold = 0.5
    size = (2, 2)
    stride = (2, 2)

    expected = [
        # tp, fp, tn, fn
        [(1, 3, 0, 0), (0, 2, 2, 0)],
        [(0, 2, 2, 0), (0, 1, 3, 0)],
    ]
    _check_window_measures(pred, gt, threshold, size, stride, expected)


def test_winperf_measures_cross():

    pred = numpy.zeros((5, 5), dtype=float)
    pred[2, :] = 1.0
    pred[:, 2] = 1.0
    pred[2, 2] = 0.0  # make one mistake at the center of the cross
    gt = numpy.zeros((5, 5), dtype=bool)
    gt[2, :] = 1.0
    gt[:, 2] = 1.0  # white cross pattern
    threshold = 0.5
    size = (3, 3)
    stride = (1, 1)

    expected = [
        # tp, fp, tn, fn
        [(4, 0, 4, 1), (4, 0, 4, 1), (4, 0, 4, 1)],
        [(4, 0, 4, 1), (4, 0, 4, 1), (4, 0, 4, 1)],
        [(4, 0, 4, 1), (4, 0, 4, 1), (4, 0, 4, 1)],
    ]
    _check_window_measures(pred, gt, threshold, size, stride, expected)


def test_winperf_measures_cross_with_padding():

    pred = numpy.zeros((5, 5), dtype=float)
    gt = numpy.zeros((5, 5), dtype=bool)
    gt[2, :] = 1.0
    gt[:, 2] = 1.0  # white cross pattern
    threshold = 0.5
    size = (4, 4)
    stride = (2, 2)

    expected = [
        # tp, fp, tn, fn
        [(0, 0, 9, 7), (0, 0, 10, 6)],
        [(0, 0, 10, 6), (0, 0, 11, 5)],
    ]
    _check_window_measures(pred, gt, threshold, size, stride, expected)


def test_winperf_measures_cross_with_padding_2():

    pred = numpy.zeros((5, 5), dtype=float)
    pred[2, :] = 1.0
    pred[:, 2] = 1.0
    pred[2, 2] = 0.0  # make one mistake at the center of the cross
    gt = numpy.zeros((5, 5), dtype=bool)
    gt[2, :] = 1.0
    gt[:, 2] = 1.0  # white cross pattern
    threshold = 0.5
    size = (4, 4)
    stride = (2, 2)

    expected = [
        # tp, fp, tn, fn
        [(6, 0, 9, 1), (5, 0, 10, 1)],
        [(5, 0, 10, 1), (4, 0, 11, 1)],
    ]
    _check_window_measures(pred, gt, threshold, size, stride, expected)


def _check_performance_summary(pred, gt, threshold, size, stride, s, figure):

    figsize = pred.shape
    pred = torch.tensor(pred)
    gt = torch.tensor(gt)

    # notice _winperf_measures() was previously tested (above)
    measures = _winperf_measures(pred, gt, threshold, size, stride)

    n_actual, avg_actual, std_actual = _performance_summary(
        figsize, measures, size, stride, figure
    )

    # the following code is not optimal, but easier to debug than the
    # list comprehension versions:
    # n_expected = numpy.array([len(k) for j in s for k in j]).reshape(figsize)
    # avg_expected = numpy.array([measures.iloc[k][figure].mean() for j in s for k in j]).reshape(figsize)
    # std_expected = numpy.array([measures.iloc[k][figure].std(ddof=1) for j in s for k in j]).reshape(figsize)
    n_expected = numpy.zeros_like(n_actual)
    avg_expected = numpy.zeros_like(avg_actual)
    std_expected = numpy.zeros_like(std_actual)
    figindex = PERFORMANCE_FIGURES.index(figure)
    for y, row in enumerate(s):
        for x, cell in enumerate(row):
            n_expected[y, x] = len(cell)
            entries = tuple(numpy.array(cell).T)  # convert indexing to numpy
            avg_expected[y, x] = measures[figindex][entries].mean()
            std_expected[y, x] = measures[figindex][entries].std(ddof=1)
    std_expected = numpy.nan_to_num(std_expected)

    assert (n_actual == n_expected).all(), (
        f"Actual N output:\n{n_actual}\n "
        f"**!=** Expected N output:\n{n_expected}"
    )

    assert (avg_actual == avg_expected).all(), (
        f"Actual average output:\n{avg_actual}\n "
        f"**!=** Expected average output:\n{avg_expected}"
    )

    assert (std_actual == std_expected).all(), (
        f"Actual std.deviation output:\n{std_actual}\n "
        f"**!=** Expected std.deviation output:\n{std_expected}"
    )


def test_performance_summary_alltrue_accuracy():

    pred = numpy.ones((4, 4), dtype=float)
    gt = numpy.ones((4, 4), dtype=bool)
    threshold = 0.5
    size = (2, 2)
    stride = (1, 1)

    # what we expect will happen for the accumulation of statistics
    # each number represents the pandas dataframe index in ``measures``
    # that needs to be accumulated for that particular pixel in the
    # original image
    stats = [
        # first row of image
        [[(0, 0)], [(0, 0), (0, 1)], [(0, 1), (0, 2)], [(0, 2)]],
        # second row of image
        [
            [(0, 0), (1, 0)],
            [(0, 0), (0, 1), (1, 0), (1, 1)],
            [(0, 1), (0, 2), (1, 1), (1, 2)],
            [(0, 2), (1, 2)],
        ],
        # third row of image
        [
            [(1, 0), (2, 0)],
            [(1, 0), (1, 1), (2, 0), (2, 1)],
            [(1, 1), (1, 2), (2, 1), (2, 2)],
            [(1, 2), (2, 2)],
        ],
        # fourth row of image
        [[(2, 0)], [(2, 0), (2, 1)], [(2, 1), (2, 2)], [(2, 2)]],
    ]

    _check_performance_summary(
        pred, gt, threshold, size, stride, stats, "accuracy"
    )
