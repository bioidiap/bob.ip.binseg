#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import numpy as np
from bob.ip.binseg.engine.inferencer import batch_metrics
import random
import shutil, tempfile
import logging
import torch

class Tester(unittest.TestCase):
    """
    Unit test for batch metrics
    """
    def setUp(self):
        self.tp = random.randint(1, 100)
        self.fp = random.randint(1, 100)
        self.tn = random.randint(1, 100)
        self.fn = random.randint(1, 100)
        self.predictions = torch.rand(size=(2,1,420,420))
        self.ground_truths = torch.randint(low=0, high=2, size=(2,1,420,420))
        self.names = ['Bob','Tim'] 
        self.output_folder = tempfile.mkdtemp()
        self.logger = logging.getLogger(__name__)

    def tearDown(self):
        # Remove the temporary folder after the test
        shutil.rmtree(self.output_folder)
    
    def test_batch_metrics(self):
        bm = batch_metrics(self.predictions, self.ground_truths, self.names, self.output_folder, self.logger)
        self.assertEqual(len(bm),2*100)
        for metric in bm:
            # check whether f1 score agree
            self.assertAlmostEqual(metric[-1],2*(metric[-6]*metric[-5])/(metric[-6]+metric[-5]))

if __name__ == '__main__':
    unittest.main()