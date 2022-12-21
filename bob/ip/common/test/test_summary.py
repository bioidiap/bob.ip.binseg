#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from ...binseg.models.driu import driu
from ...binseg.models.driu_od import driu_od
from ...binseg.models.hed import hed
from ...binseg.models.resunet import resunet50
from ...binseg.models.unet import unet
from ...detect.models.faster_rcnn import faster_rcnn
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

    def test_summary_fasterrcnn(self):
        model = faster_rcnn()
        s, param = summary(model)
        self.assertIsInstance(s, str)
        self.assertIsInstance(param, int)