#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import unittest

import math
import torch
import nose.tools

from ..utils.measure import base_measures, auc
from ..engine.evaluator import sample_measures_for_threshold


class Tester(unittest.TestCase):
    """
    Unit test for base measures
    """

    def setUp(self):
        self.tp = random.randint(1, 100)
        self.fp = random.randint(1, 100)
        self.tn = random.randint(1, 100)
        self.fn = random.randint(1, 100)

    def test_precision(self):
        precision = base_measures(self.tp, self.fp, self.tn, self.fn)[0]
        self.assertEqual((self.tp) / (self.tp + self.fp), precision)

    def test_recall(self):
        recall = base_measures(self.tp, self.fp, self.tn, self.fn)[1]
        self.assertEqual((self.tp) / (self.tp + self.fn), recall)

    def test_specificity(self):
        specificity = base_measures(self.tp, self.fp, self.tn, self.fn)[2]
        self.assertEqual((self.tn) / (self.tn + self.fp), specificity)

    def test_accuracy(self):
        accuracy = base_measures(self.tp, self.fp, self.tn, self.fn)[3]
        self.assertEqual(
            (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn),
            accuracy,
        )

    def test_jaccard(self):
        jaccard = base_measures(self.tp, self.fp, self.tn, self.fn)[4]
        self.assertEqual(self.tp / (self.tp + self.fp + self.fn), jaccard)

    def test_f1(self):
        p, r, s, a, j, f1 = base_measures(self.tp, self.fp, self.tn, self.fn)
        self.assertEqual(
            (2.0 * self.tp) / (2.0 * self.tp + self.fp + self.fn), f1
        )
        self.assertAlmostEqual((2 * p * r) / (p + r), f1)  # base definition


def test_auc():

    # basic tests
    assert math.isclose(auc([0.0, 0.5, 1.0], [1.0, 1.0, 1.0]), 1.0)
    assert math.isclose(
        auc([0.0, 0.5, 1.0], [1.0, 0.5, 0.0]), 0.5, rel_tol=0.001
    )
    assert math.isclose(
        auc([0.0, 0.5, 1.0], [0.0, 0.0, 0.0]), 0.0, rel_tol=0.001
    )
    assert math.isclose(
        auc([0.0, 0.5, 1.0], [0.0, 1.0, 0.0]), 0.5, rel_tol=0.001
    )
    assert math.isclose(
        auc([0.0, 0.5, 1.0], [0.0, 0.5, 0.0]), 0.25, rel_tol=0.001
    )
    assert math.isclose(
        auc([0.0, 0.5, 1.0], [0.0, 0.5, 0.0]), 0.25, rel_tol=0.001
    )

    # reversing tht is also true
    assert math.isclose(auc([0.0, 0.5, 1.0][::-1], [1.0, 1.0, 1.0][::-1]), 1.0)
    assert math.isclose(
        auc([0.0, 0.5, 1.0][::-1], [1.0, 0.5, 0.0][::-1]), 0.5, rel_tol=0.001
    )
    assert math.isclose(
        auc([0.0, 0.5, 1.0][::-1], [0.0, 0.0, 0.0][::-1]), 0.0, rel_tol=0.001
    )
    assert math.isclose(
        auc([0.0, 0.5, 1.0][::-1], [0.0, 1.0, 0.0][::-1]), 0.5, rel_tol=0.001
    )
    assert math.isclose(
        auc([0.0, 0.5, 1.0][::-1], [0.0, 0.5, 0.0][::-1]), 0.25, rel_tol=0.001
    )
    assert math.isclose(
        auc([0.0, 0.5, 1.0][::-1], [0.0, 0.5, 0.0][::-1]), 0.25, rel_tol=0.001
    )


@nose.tools.raises(ValueError)
def test_auc_raises_value_error():

    # x is **not** monotonically increasing or decreasing
    assert math.isclose(auc([0.0, 0.5, 0.0], [1.0, 1.0, 1.0]), 1.0)


@nose.tools.raises(AssertionError)
def test_auc_raises_assertion_error():

    # x is **not** the same size as y
    assert math.isclose(auc([0.0, 0.5, 1.0], [1.0, 1.0]), 1.0)


def test_sample_measures_mask_checkerbox():

    prediction = torch.ones((4, 4), dtype=float)
    ground_truth = torch.ones((4, 4), dtype=float)
    ground_truth[2:, :2] = 0.0
    ground_truth[:2, 2:] = 0.0
    mask = torch.zeros((4, 4), dtype=float)
    mask[1:3, 1:3] = 1.0
    threshold = 0.5

    # with this configuration, this should be the correct count
    tp = 2
    fp = 2
    tn = 0
    fn = 0

    nose.tools.eq_(
        base_measures(tp, fp, tn, fn),
        sample_measures_for_threshold(
            prediction, ground_truth, mask, threshold
        ),
    )


def test_sample_measures_mask_cross():

    prediction = torch.ones((10, 10), dtype=float)
    prediction[0,:] = 0.0
    prediction[9,:] = 0.0
    ground_truth = torch.ones((10, 10), dtype=float)
    ground_truth[:5,] = 0.0  #lower part is not to be set
    mask = torch.zeros((10, 10), dtype=float)
    mask[(0,1,2,3,4,5,6,7,8,9),(0,1,2,3,4,5,6,7,8,9)] = 1.0
    mask[(0,1,2,3,4,5,6,7,8,9),(9,8,7,6,5,4,3,2,1,0)] = 1.0
    threshold = 0.5

    # with this configuration, this should be the correct count
    tp = 8
    fp = 8
    tn = 2
    fn = 2

    nose.tools.eq_(
        base_measures(tp, fp, tn, fn),
        sample_measures_for_threshold(
            prediction, ground_truth, mask, threshold
        ),
    )


def test_sample_measures_mask_border():

    prediction = torch.zeros((10, 10), dtype=float)
    prediction[:,4] = 1.0
    prediction[:,5] = 1.0
    prediction[0,4] = 0.0
    prediction[8,4] = 0.0
    prediction[1,6] = 1.0
    ground_truth = torch.zeros((10, 10), dtype=float)
    ground_truth[:,4] = 1.0
    ground_truth[:,5] = 1.0
    mask = torch.ones((10, 10), dtype=float)
    mask[:,0] = 0.0
    mask[0,:] = 0.0
    mask[:,9] = 0.0
    mask[9,:] = 0.0
    threshold = 0.5

    # with this configuration, this should be the correct count
    tp = 15
    fp = 1
    tn = 47
    fn = 1

    nose.tools.eq_(
        base_measures(tp, fp, tn, fn),
        sample_measures_for_threshold(
            prediction, ground_truth, mask, threshold
        ),
    )
