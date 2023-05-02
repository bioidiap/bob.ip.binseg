# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import unittest

from collections import OrderedDict
from tempfile import TemporaryDirectory

import torch

from deepdraw.utils.checkpointer import Checkpointer


class TestCheckpointer(unittest.TestCase):
    def create_model(self):
        return torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))

    def create_complex_model(self):
        m = torch.nn.Module()
        m.block1 = torch.nn.Module()
        m.block1.layer1 = torch.nn.Linear(2, 3)
        m.layer2 = torch.nn.Linear(3, 2)
        m.res = torch.nn.Module()
        m.res.layer2 = torch.nn.Linear(3, 2)

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
            assert fresh_checkpointer.last_checkpoint() == os.path.realpath(
                os.path.join(f, "checkpoint_file.pth")
            )
            _ = fresh_checkpointer.load()

        for trained_p, loaded_p in zip(
            trained_model.parameters(), fresh_model.parameters()
        ):
            # different tensor references
            assert id(trained_p) != id(loaded_p)
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
                assert fresh_checkpointer.last_checkpoint() is None
                _ = fresh_checkpointer.load(
                    os.path.join(f, "checkpoint_file.pth")
                )

        for trained_p, loaded_p in zip(
            trained_model.parameters(), fresh_model.parameters()
        ):
            # different tensor references
            assert id(trained_p) != id(loaded_p)
            # same content
            assert trained_p.equal(loaded_p)

    def test_checkpointer_process(self):
        from deepdraw.configs.models.lwnet import model
        from deepdraw.engine.trainer import checkpointer_process

        with TemporaryDirectory() as f:
            checkpointer = Checkpointer(model, path=f)
            lowest_validation_loss = 0.001
            checkpoint_period = 0
            valid_loss = 1.5
            arguments = {"epoch": 0, "max_epoch": 10}
            epoch = 1
            max_epoch = 10
            lowest_validation_loss = checkpointer_process(
                checkpointer,
                checkpoint_period,
                valid_loss,
                lowest_validation_loss,
                arguments,
                epoch,
                max_epoch,
            )

            assert lowest_validation_loss == 0.001

            lowest_validation_loss = 1000
            lowest_validation_loss = checkpointer_process(
                checkpointer,
                checkpoint_period,
                valid_loss,
                lowest_validation_loss,
                arguments,
                epoch,
                max_epoch,
            )

            assert lowest_validation_loss == 1.5
