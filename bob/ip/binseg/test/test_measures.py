#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import unittest

import math
import nose.tools

from ..utils.measure import base_measures, auc


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
