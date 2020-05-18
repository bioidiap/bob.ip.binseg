#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

from ..models.driu import driu
from ..models.driu_od import driu_od
from ..models.hed import hed
from ..models.unet import unet
from ..models.resunet import resunet50
from ..utils.summary import summary


class Tester(unittest.TestCase):
    """
    Unit test for model architectures
    """

    def test_summary_driu(self):
        model = driu()
        s, param = summary(model)
        self.assertIsInstance(s, str)
        self.assertIsInstance(param, int)

    def test_summary_driuod(self):
        model = driu_od()
        s, param = summary(model)
        self.assertIsInstance(s, str)
        self.assertIsInstance(param, int)

    def test_summary_hed(self):
        model = hed()
        s, param = summary(model)
        self.assertIsInstance(s, str)
        self.assertIsInstance(param, int)

    def test_summary_unet(self):
        model = unet()
        s, param = summary(model)
        self.assertIsInstance(s, str)
        self.assertIsInstance(param, int)

    def test_summary_resunet(self):
        model = resunet50()
        s, param = summary(model)
        self.assertIsInstance(s, str)
        self.assertIsInstance(param, int)
