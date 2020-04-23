#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import random
import shutil

import torch
import pandas
import numpy

from ..engine.evaluator import _sample_metrics

import logging
logger = logging.getLogger(__name__)


class Tester(unittest.TestCase):
    """
    Unit test for batch metrics
    """

    def setUp(self):
        self.tp = random.randint(1, 100)
        self.fp = random.randint(1, 100)
        self.tn = random.randint(1, 100)
        self.fn = random.randint(1, 100)
        self.predictions = torch.rand(size=(2, 1, 420, 420))
        self.ground_truths = torch.randint(low=0, high=2, size=(2, 1, 420, 420))
        self.names = ["Bob", "Tim"]

    def test_batch_metrics(self):
        dfs = []
        for pred, gt in zip(self.predictions, self.ground_truths):
            dfs.append(_sample_metrics(pred, gt, 100))
        bm = pandas.concat(dfs)

        self.assertEqual(len(bm), 2 * 100)
        # check whether f1 score agree
        calculated = bm.f1_score.to_numpy()
        ours = (2*(bm.precision*bm.recall)/(bm.precision+bm.recall)).to_numpy()
        assert numpy.isclose(calculated, ours).all()


if __name__ == "__main__":
    unittest.main()
