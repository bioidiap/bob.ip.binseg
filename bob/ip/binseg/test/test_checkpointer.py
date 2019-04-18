#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/engine/trainer.py 
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import OrderedDict
from tempfile import TemporaryDirectory
import unittest

import torch
from torch import nn
import os

from bob.ip.binseg.utils.checkpointer import Checkpointer


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
            checkpointer = Checkpointer(
                trained_model, save_dir=f, save_to_disk=True
            )
            checkpointer.save("checkpoint_file")

            # in the same folder
            fresh_checkpointer = Checkpointer(fresh_model, save_dir=f)
            self.assertTrue(fresh_checkpointer.has_checkpoint())
            self.assertEqual(
                fresh_checkpointer.get_checkpoint_file(),
                os.path.join(f, "checkpoint_file.pth"),
            )
            _ = fresh_checkpointer.load()

        for trained_p, loaded_p in zip(
            trained_model.parameters(), fresh_model.parameters()
        ):
            # different tensor references
            self.assertFalse(id(trained_p) == id(loaded_p))
            # same content
            self.assertTrue(trained_p.equal(loaded_p))

    def test_from_name_file_model(self):
        # test that loading works even if they differ by a prefix
        trained_model = self.create_model()
        fresh_model = self.create_model()
        with TemporaryDirectory() as f:
            checkpointer = Checkpointer(
                trained_model, save_dir=f, save_to_disk=True
            )
            checkpointer.save("checkpoint_file")

            # on different folders
            with TemporaryDirectory() as g:
                fresh_checkpointer = Checkpointer(fresh_model, save_dir=g)
                self.assertFalse(fresh_checkpointer.has_checkpoint())
                self.assertEqual(fresh_checkpointer.get_checkpoint_file(), "")
                _ = fresh_checkpointer.load(os.path.join(f, "checkpoint_file.pth"))

        for trained_p, loaded_p in zip(
            trained_model.parameters(), fresh_model.parameters()
        ):
            # different tensor references
            self.assertFalse(id(trained_p) == id(loaded_p))
            # same content
            self.assertTrue(trained_p.equal(loaded_p))


if __name__ == "__main__":
    unittest.main()