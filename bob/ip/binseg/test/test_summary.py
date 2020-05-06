#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

from ..modeling.driu import build_driu
from ..modeling.driuod import build_driuod
from ..modeling.hed import build_hed
from ..modeling.unet import build_unet
from ..modeling.resunet import build_res50unet
from ..utils.summary import summary


class Tester(unittest.TestCase):
    """
    Unit test for model architectures
    """

    def test_summary_driu(self):
        model = build_driu()
        s, param = summary(model)
        self.assertIsInstance(s, str)
        self.assertIsInstance(param, int)

    def test_summary_driuod(self):
        model = build_driuod()
        s, param = summary(model)
        self.assertIsInstance(s, str)
        self.assertIsInstance(param, int)

    def test_summary_hed(self):
        model = build_hed()
        s, param = summary(model)
        self.assertIsInstance(s, str)
        self.assertIsInstance(param, int)

    def test_summary_unet(self):
        model = build_unet()
        s, param = summary(model)
        self.assertIsInstance(s, str)
        self.assertIsInstance(param, int)

    def test_summary_resunet(self):
        model = build_res50unet()
        s, param = summary(model)
        self.assertIsInstance(s, str)
        self.assertIsInstance(param, int)
