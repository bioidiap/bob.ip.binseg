#!/usr/bin/env python
# coding=utf-8

import nose.tools
from nose.plugins.attrib import attr

import torch

from . import mock_dataset
stare_dataset, stare_variable_set = mock_dataset()
from .utils import rc_variable_set


@rc_variable_set("bob.ip.binseg.drive.datadir")
def test_drive_default_train():

    from ..configs.datasets.drive import dataset
    nose.tools.eq_(len(dataset), 20)
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
    nose.tools.eq_(len(dataset), 20)
    for sample in dataset:
        nose.tools.eq_(len(sample), 4)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 544, 544)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 544, 544)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)
        nose.tools.eq_(sample[3].shape, (1, 544, 544)) #planes, height, width
        nose.tools.eq_(sample[3].dtype, torch.float32)


@stare_variable_set("bob.ip.binseg.stare.datadir")
def test_stare_default_train():

    from ..configs.datasets.stare import dataset
    # hack to allow testing on the CI
    dataset._samples = stare_dataset.subsets("default")["train"]
    nose.tools.eq_(len(dataset), 10)
    for sample in dataset:
        nose.tools.eq_(len(sample), 3)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 608, 704)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 608, 704)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)


@stare_variable_set("bob.ip.binseg.stare.datadir")
def test_stare_default_test():

    from ..configs.datasets.stare_test import dataset
    # hack to allow testing on the CI
    dataset._samples = stare_dataset.subsets("default")["test"]
    nose.tools.eq_(len(dataset), 10)
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
    nose.tools.eq_(len(dataset), 8)
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
    nose.tools.eq_(len(dataset), 20)
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
    nose.tools.eq_(len(dataset), 15)
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
    nose.tools.eq_(len(dataset), 30)
    for sample in dataset:
        nose.tools.eq_(len(sample), 4)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 1168, 1648)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 1168, 1648)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)
        nose.tools.eq_(sample[3].shape, (1, 1168, 1648)) #planes, height, width
        nose.tools.eq_(sample[3].dtype, torch.float32)


@rc_variable_set("bob.ip.binseg.refuge.datadir")
@attr("slow")
def test_refuge_optic_disc_train():

    from ..configs.datasets.refuge_od import dataset
    nose.tools.eq_(len(dataset), 400)
    for sample in dataset:
        nose.tools.eq_(len(sample), 3)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 1632, 1632)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 1632, 1632)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)


@rc_variable_set("bob.ip.binseg.refuge.datadir")
@attr("slow")
def test_refuge_optic_disc_dev():

    from ..configs.datasets.refuge_od_dev import dataset
    nose.tools.eq_(len(dataset), 400)
    for sample in dataset:
        nose.tools.eq_(len(sample), 3)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 1632, 1632)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 1632, 1632)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)


@rc_variable_set("bob.ip.binseg.refuge.datadir")
@attr("slow")
def test_refuge_optic_disc_test():

    from ..configs.datasets.refuge_od_test import dataset
    nose.tools.eq_(len(dataset), 400)
    for sample in dataset:
        nose.tools.eq_(len(sample), 3)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 1632, 1632)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 1632, 1632)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)


@rc_variable_set("bob.ip.binseg.refuge.datadir")
@attr("slow")
def test_refuge_optic_cup_train():

    from ..configs.datasets.refuge_cup import dataset
    nose.tools.eq_(len(dataset), 400)
    for sample in dataset:
        nose.tools.eq_(len(sample), 3)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 1632, 1632)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 1632, 1632)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)


@rc_variable_set("bob.ip.binseg.refuge.datadir")
@attr("slow")
def test_refuge_optic_cup_dev():

    from ..configs.datasets.refuge_cup_dev import dataset
    nose.tools.eq_(len(dataset), 400)
    for sample in dataset:
        nose.tools.eq_(len(sample), 3)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 1632, 1632)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 1632, 1632)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)


@rc_variable_set("bob.ip.binseg.refuge.datadir")
@attr("slow")
def test_refuge_optic_cup_test():

    from ..configs.datasets.refuge_cup_test import dataset
    nose.tools.eq_(len(dataset), 400)
    for sample in dataset:
        nose.tools.eq_(len(sample), 3)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 1632, 1632)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 1632, 1632)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)


@rc_variable_set("bob.ip.binseg.drishtigs1.datadir")
def test_drishtigs1_optic_disc_all_train():

    from ..configs.datasets.dristhigs1_od import dataset
    nose.tools.eq_(len(dataset), 50)
    for sample in dataset:
        nose.tools.eq_(len(sample), 3)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 1760, 2048)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 1760, 2048)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)


@rc_variable_set("bob.ip.binseg.drishtigs1.datadir")
def test_drishtigs1_optic_disc_all_test():

    from ..configs.datasets.dristhigs1_od_test import dataset
    nose.tools.eq_(len(dataset), 51)
    for sample in dataset:
        nose.tools.eq_(len(sample), 3)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 1760, 2048)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 1760, 2048)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)


@rc_variable_set("bob.ip.binseg.drishtigs1.datadir")
def test_drishtigs1_optic_cup_all_train():

    from ..configs.datasets.dristhigs1_cup import dataset
    nose.tools.eq_(len(dataset), 50)
    for sample in dataset:
        nose.tools.eq_(len(sample), 3)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 1760, 2048)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 1760, 2048)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)


@rc_variable_set("bob.ip.binseg.drishtigs1.datadir")
def test_drishtigs1_optic_cup_all_test():

    from ..configs.datasets.dristhigs1_cup_test import dataset
    nose.tools.eq_(len(dataset), 51)
    for sample in dataset:
        nose.tools.eq_(len(sample), 3)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 1760, 2048)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 1760, 2048)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)


@rc_variable_set("bob.ip.binseg.drionsdb.datadir")
def test_drionsdb_default_train():

    from ..configs.datasets.drionsdb import dataset
    nose.tools.eq_(len(dataset), 60)
    for sample in dataset:
        nose.tools.eq_(len(sample), 3)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 416, 608)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 416, 608)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)


@rc_variable_set("bob.ip.binseg.drionsdb.datadir")
def test_drionsdb_default_test():

    from ..configs.datasets.drionsdb_test import dataset
    nose.tools.eq_(len(dataset), 50)
    for sample in dataset:
        nose.tools.eq_(len(sample), 3)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 416, 608)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 416, 608)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)
