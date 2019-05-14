#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import unittest
import numpy as np
from bob.ip.binseg.modeling.driu import build_driu
from bob.ip.binseg.modeling.driuod import build_driuod
from bob.ip.binseg.modeling.hed import build_hed
from bob.ip.binseg.modeling.unet import build_unet
from bob.ip.binseg.modeling.resunet import build_res50unet
from bob.ip.binseg.utils.summary import summary

class Tester(unittest.TestCase):
    """
    Unit test for model architectures
    """    
    def test_summary_driu(self):
        model = build_driu()
        param = summary(model)
        self.assertIsInstance(param,int)


    def test__summary_driuod(self):
        model = build_driuod()
        param = summary(model)
        self.assertIsInstance(param,int)


    def test_summary_hed(self):
        model = build_hed()
        param = summary(model)
        self.assertIsInstance(param,int)

    def test_summary_unet(self):
        model = build_unet()
        param = summary(model)
        self.assertIsInstance(param,int)

    def test_summary_resunet(self):
        model = build_res50unet()
        param = summary(model)
        self.assertIsInstance(param,int)

if __name__ == '__main__':
    unittest.main()