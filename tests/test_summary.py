# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import unittest

from deepdraw.binseg.models.driu import driu
from deepdraw.binseg.models.driu_od import driu_od
from deepdraw.binseg.models.hed import hed
from deepdraw.binseg.models.mean_teacher import mean_teacher
from deepdraw.binseg.models.resunet import resunet50
from deepdraw.binseg.models.unet import unet
from deepdraw.common.utils.summary import summary
from deepdraw.detect.models.faster_rcnn import faster_rcnn


class Tester(unittest.TestCase):
    """Unit test for model architectures."""

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

    def test_summary_mean_teacher(self):
        weight = None
        model = mean_teacher(weight)
        s, param = summary(model)
        self.assertIsInstance(s, str)
        self.assertIsInstance(param, int)
