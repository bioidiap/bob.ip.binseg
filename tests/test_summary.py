# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import unittest

from deepdraw.models.driu import driu
from deepdraw.models.driu_od import driu_od
from deepdraw.models.hed import hed
from deepdraw.models.resunet import resunet50
from deepdraw.models.unet import unet
from deepdraw.utils.summary import summary


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
