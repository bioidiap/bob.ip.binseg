#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest
from collections import OrderedDict
from tempfile import TemporaryDirectory

import torch
import nose.tools
from torch import nn

from ..utils.checkpointer import Checkpointer


class TestCheckpointer(unittest.TestCase):
    def create_model(self):
        return nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))

    def create_complex_model(self):
        m = nn.Module()
        m.block1 = nn.Module()
        m.block1.layer1 = nn.Linear(2, 3)
        m.layer2 = nn.Linear(3, 2)
        m.res = nn.Module()
        m.res.layer2 = nn.Linear(3, 2)

        state_dict = OrderedDict()
        state_dict["layer1.weight"] = torch.rand(3, 2)
        state_dict["layer1.bias"] = torch.rand(3)
        state_dict["layer2.weight"] = torch.rand(2, 3)
        state_dict["layer2.bias"] = torch.rand(2)
        state_dict["res.layer2.weight"] = torch.rand(2, 3)
        state_dict["res.layer2.bias"] = torch.rand(2)

        return m, state_dict

    def test_from_last_checkpoint_model(self):
        # test that loading works even if they differ by a prefix
        trained_model = self.create_model()
        fresh_model = self.create_model()
        with TemporaryDirectory() as f:
            checkpointer = Checkpointer(trained_model, path=f)
            checkpointer.save("checkpoint_file")

            # in the same folder
            fresh_checkpointer = Checkpointer(fresh_model, path=f)
            assert fresh_checkpointer.has_checkpoint()
            nose.tools.eq_(fresh_checkpointer.last_checkpoint(),
                    os.path.realpath(os.path.join(f, "checkpoint_file.pth")))
            _ = fresh_checkpointer.load()

        for trained_p, loaded_p in zip(
            trained_model.parameters(), fresh_model.parameters()
        ):
            # different tensor references
            nose.tools.assert_not_equal(id(trained_p), id(loaded_p))
            # same content
            assert trained_p.equal(loaded_p)

    def test_from_name_file_model(self):
        # test that loading works even if they differ by a prefix
        trained_model = self.create_model()
        fresh_model = self.create_model()
        with TemporaryDirectory() as f:
            checkpointer = Checkpointer(trained_model, path=f)
            checkpointer.save("checkpoint_file")

            # on different folders
            with TemporaryDirectory() as g:
                fresh_checkpointer = Checkpointer(fresh_model, path=g)
                assert not fresh_checkpointer.has_checkpoint()
                nose.tools.eq_(fresh_checkpointer.last_checkpoint(), None)
                _ = fresh_checkpointer.load(os.path.join(f, "checkpoint_file.pth"))

        for trained_p, loaded_p in zip(
            trained_model.parameters(), fresh_model.parameters()
        ):
            # different tensor references
            nose.tools.assert_not_equal(id(trained_p), id(loaded_p))
            # same content
            assert trained_p.equal(loaded_p)
