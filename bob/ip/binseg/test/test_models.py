#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import unittest
import numpy as np
from bob.ip.binseg.modeling.driu import build_driu
from bob.ip.binseg.modeling.hed import build_hed

class Tester(unittest.TestCase):
    """
    Unit test for model architectures
    """
    x = torch.randn(1, 3, 544, 544)
    hw = np.array(x.shape)[[2,3]]
    
    def test_driu(self):
        model = build_driu()
        out = model(Tester.x)
        out_hw = np.array(out.shape)[[2,3]]
        self.assertEqual(Tester.hw.all(), out_hw.all())


    def test_hed(self):
        model = build_hed()
        out = model(Tester.x)
        # NOTE: HED outputs a list of length 4. We test only for the last concat-fuse layer
        out_hw = np.array(out[4].shape)[[2,3]]
        self.assertEqual(Tester.hw.all(), out_hw.all())


if __name__ == '__main__':
    unittest.main()