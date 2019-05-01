#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import numpy as np
from bob.ip.binseg.utils.metric import base_metrics
import random

class Tester(unittest.TestCase):
    """
    Unit test for base metrics
    """
    def setUp(self):
        self.tp = random.randint(1, 100)
        self.fp = random.randint(1, 100)
        self.tn = random.randint(1, 100)
        self.fn = random.randint(1, 100)
    
    def test_precision(self):
        precision = base_metrics(self.tp, self.fp, self.tn, self.fn)[0]
        self.assertEqual((self.tp)/(self.tp + self.fp),precision)

    def test_recall(self):
        recall = base_metrics(self.tp, self.fp, self.tn, self.fn)[1]
        self.assertEqual((self.tp)/(self.tp + self.fn),recall)

    def test_specificity(self):
        specificity = base_metrics(self.tp, self.fp, self.tn, self.fn)[2]
        self.assertEqual((self.tn)/(self.tn + self.fp),specificity)
    
    def test_accuracy(self):
        accuracy = base_metrics(self.tp, self.fp, self.tn, self.fn)[3]
        self.assertEqual((self.tp + self.tn)/(self.tp + self.tn + self.fp + self.fn), accuracy)

    def test_jaccard(self):
        jaccard = base_metrics(self.tp, self.fp, self.tn, self.fn)[4]
        self.assertEqual(self.tp / (self.tp+self.fp+self.fn), jaccard)

    def test_f1(self):
        f1 = base_metrics(self.tp, self.fp, self.tn, self.fn)[5]
        self.assertEqual((2.0 * self.tp ) / (2.0 * self.tp + self.fp + self.fn ),f1)
        
if __name__ == '__main__':
    unittest.main()