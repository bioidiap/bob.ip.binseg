#!/usr/bin/env python
# coding=utf-8

import nose.tools

import torch

from .utils import rc_variable_set


@rc_variable_set("bob.ip.binseg.drive.datadir")
def test_drive_default_train():

    from ..configs.datasets.drive import dataset
    for sample in dataset:
        nose.tools.eq_(len(sample), 4)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 544, 544)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 544, 544)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)
        nose.tools.eq_(sample[3].shape, (1, 544, 544)) #planes, height, width
        nose.tools.eq_(sample[3].dtype, torch.float32)


@rc_variable_set("bob.ip.binseg.drive.datadir")
def test_drive_default_test():

    from ..configs.datasets.drive_test import dataset
    for sample in dataset:
        nose.tools.eq_(len(sample), 4)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 544, 544)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 544, 544)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)
        nose.tools.eq_(sample[3].shape, (1, 544, 544)) #planes, height, width
        nose.tools.eq_(sample[3].dtype, torch.float32)


@rc_variable_set("bob.ip.binseg.stare.datadir")
def test_stare_default_train():

    from ..configs.datasets.stare import dataset
    for sample in dataset:
        nose.tools.eq_(len(sample), 3)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 608, 704)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 608, 704)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)


@rc_variable_set("bob.ip.binseg.stare.datadir")
def test_stare_default_test():

    from ..configs.datasets.stare_test import dataset
    for sample in dataset:
        nose.tools.eq_(len(sample), 3)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 608, 704)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 608, 704)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)


@rc_variable_set("bob.ip.binseg.chasedb1.datadir")
def test_chasedb1_default_train():

    from ..configs.datasets.chasedb1 import dataset
    for sample in dataset:
        nose.tools.eq_(len(sample), 3)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 960, 960)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 960, 960)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)


@rc_variable_set("bob.ip.binseg.chasedb1.datadir")
def test_chasedb1_default_test():

    from ..configs.datasets.chasedb1_test import dataset
    for sample in dataset:
        nose.tools.eq_(len(sample), 3)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 960, 960)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 960, 960)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)


@rc_variable_set("bob.ip.binseg.hrf.datadir")
def test_hrf_default_train():

    from ..configs.datasets.hrf_1168 import dataset
    for sample in dataset:
        nose.tools.eq_(len(sample), 4)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 1168, 1648)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 1168, 1648)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)
        nose.tools.eq_(sample[3].shape, (1, 1168, 1648)) #planes, height, width
        nose.tools.eq_(sample[3].dtype, torch.float32)


@rc_variable_set("bob.ip.binseg.hrf.datadir")
def test_hrf_default_test():

    from ..configs.datasets.hrf_1168_test import dataset
    for sample in dataset:
        nose.tools.eq_(len(sample), 4)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 1168, 1648)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 1168, 1648)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)
        nose.tools.eq_(sample[3].shape, (1, 1168, 1648)) #planes, height, width
        nose.tools.eq_(sample[3].dtype, torch.float32)
