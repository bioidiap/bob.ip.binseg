#!/usr/bin/env python
# coding=utf-8

"""Tests for significance tools"""


import numpy
import pandas
import nose.tools
import torch

from ..engine.significance import _patch_measures
from ..utils.measure import base_measures


def _check_patch_measures(pred, gt, threshold, size, stride, expected):

    pred = torch.tensor(pred)
    gt = torch.tensor(gt)
    actual = _patch_measures(pred, gt, threshold, size, stride)

    # transforms tp,tn,fp,fn through base_measures()
    expected = pandas.DataFrame([k[:2] + base_measures(*k[2:]) for k in expected],
            columns=[
                "y",
                "x",
                "precision",  # tp/(tp+fp)
                "recall",  # tpr = tp/p = tp/(tp+fn)
                "specificity",  # tnr = tn/n = tn/(tn+fp)
                "accuracy",  # (tp+tn)/(p+n) = (tp+tn)/(tp+fn+tn+fp)
                "jaccard",  #  f1/(2-f1) = tp/(tp+fp+fn)
                "f1_score",  # 2*rp/(r+p) = 2*tp/(2*tp+fp+fn)
                ])

    assert (actual == expected).all().all(), f"Actual output:\n{actual}\n " \
            f"**!=** Expected output:\n{expected}"


def test_patch_measures_alltrue():

    pred = numpy.ones((4,4), dtype=float)
    gt = numpy.ones((4,4), dtype=bool)
    threshold = 0.5
    size = (2,2)
    stride = (1,1)

    expected = [
            #y, x, tp, fp, tn, fn
            (0, 0,  4,  0,  0,  0),
            (0, 1,  4,  0,  0,  0),
            (0, 2,  4,  0,  0,  0),
            (1, 0,  4,  0,  0,  0),
            (1, 1,  4,  0,  0,  0),
            (1, 2,  4,  0,  0,  0),
            (2, 0,  4,  0,  0,  0),
            (2, 1,  4,  0,  0,  0),
            (2, 2,  4,  0,  0,  0),
            ]
    _check_patch_measures(pred, gt, threshold, size, stride, expected)


def test_patch_measures_alltrue_with_padding():

    pred = numpy.ones((3,3), dtype=float)
    gt = numpy.ones((3,3), dtype=bool)
    threshold = 0.5
    size = (2,2)
    stride = (2,2)

    expected = [
            #y, x, tp, fp, tn, fn
            (0, 0,  4,  0,  0,  0),
            (0, 1,  2,  0,  2,  0),
            (1, 0,  2,  0,  2,  0),
            (1, 1,  1,  0,  3,  0),
            ]
    _check_patch_measures(pred, gt, threshold, size, stride, expected)


def test_patch_measures_dot_with_padding():

    pred = numpy.ones((3,3), dtype=float)
    gt = numpy.zeros((3,3), dtype=bool)
    gt[1,1] = 1.0  #white dot pattern
    threshold = 0.5
    size = (2,2)
    stride = (2,2)

    expected = [
            #y, x, tp, fp, tn, fn
            (0, 0,  1,  3,  0,  0),
            (0, 1,  0,  2,  2,  0),
            (1, 0,  0,  2,  2,  0),
            (1, 1,  0,  1,  3,  0),
            ]
    _check_patch_measures(pred, gt, threshold, size, stride, expected)


def test_patch_measures_cross():

    pred = numpy.zeros((5,5), dtype=float)
    pred[2,:] = 1.0
    pred[:,2] = 1.0
    pred[2,2] = 0.0  #make one mistake at the center of the cross
    gt = numpy.zeros((5,5), dtype=bool)
    gt[2,:] = 1.0
    gt[:,2] = 1.0  #white cross pattern
    threshold = 0.5
    size = (3,3)
    stride = (1,1)

    expected = [
            #y, x, tp, fp, tn, fn
            (0, 0,  4,  0,  4,  1),
            (0, 1,  4,  0,  4,  1),
            (0, 2,  4,  0,  4,  1),
            (1, 0,  4,  0,  4,  1),
            (1, 1,  4,  0,  4,  1),
            (1, 2,  4,  0,  4,  1),
            (2, 0,  4,  0,  4,  1),
            (2, 1,  4,  0,  4,  1),
            (2, 2,  4,  0,  4,  1),
            ]
    _check_patch_measures(pred, gt, threshold, size, stride, expected)


def test_patch_measures_cross_with_padding():

    pred = numpy.zeros((5,5), dtype=float)
    gt = numpy.zeros((5,5), dtype=bool)
    gt[2,:] = 1.0
    gt[:,2] = 1.0  #white cross pattern
    threshold = 0.5
    size = (4,4)
    stride = (2,2)

    expected = [
            #y, x, tp, fp, tn, fn
            (0, 0,  0,  0,  9,  7),
            (0, 1,  0,  0,  10,  6),
            (1, 0,  0,  0,  10,  6),
            (1, 1,  0,  0,  11,  5),
            ]
    _check_patch_measures(pred, gt, threshold, size, stride, expected)


def test_patch_measures_cross_with_padding_2():

    pred = numpy.zeros((5,5), dtype=float)
    pred[2,:] = 1.0
    pred[:,2] = 1.0
    pred[2,2] = 0.0  #make one mistake at the center of the cross
    gt = numpy.zeros((5,5), dtype=bool)
    gt[2,:] = 1.0
    gt[:,2] = 1.0  #white cross pattern
    threshold = 0.5
    size = (4,4)
    stride = (2,2)

    expected = [
            #y, x, tp, fp, tn, fn
            (0, 0,  6,  0,  9,  1),
            (0, 1,  5,  0,  10,  1),
            (1, 0,  5,  0,  10,  1),
            (1, 1,  4,  0,  11,  1),
            ]
    _check_patch_measures(pred, gt, threshold, size, stride, expected)
