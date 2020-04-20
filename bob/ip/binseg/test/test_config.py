#!/usr/bin/env python
# coding=utf-8

import nose.tools
from nose.plugins.attrib import attr

import torch

from . import mock_dataset
stare_dataset, stare_variable_set = mock_dataset()
from .utils import rc_variable_set


@rc_variable_set("bob.ip.binseg.drive.datadir")
def test_drive_default():

    from ..configs.datasets.drive.default import dataset
    nose.tools.eq_(len(dataset["train"]), 20)
    for sample in dataset["train"]:
        nose.tools.eq_(len(sample), 4)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 544, 544)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 544, 544)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)
        nose.tools.eq_(sample[3].shape, (1, 544, 544)) #planes, height, width
        nose.tools.eq_(sample[3].dtype, torch.float32)


    nose.tools.eq_(len(dataset["test"]), 20)
    for sample in dataset["test"]:
        nose.tools.eq_(len(sample), 4)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 544, 544)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 544, 544)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)
        nose.tools.eq_(sample[3].shape, (1, 544, 544)) #planes, height, width
        nose.tools.eq_(sample[3].dtype, torch.float32)


@stare_variable_set("bob.ip.binseg.stare.datadir")
def test_stare_ah():

    from ..configs.datasets.stare.ah import dataset
    # hack to allow testing on the CI
    dataset["train"]._samples = stare_dataset.subsets("ah")["train"]

    nose.tools.eq_(len(dataset["train"]), 10)
    for sample in dataset["train"]:
        nose.tools.eq_(len(sample), 3)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 608, 704)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 608, 704)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)

    # hack to allow testing on the CI
    dataset["test"]._samples = stare_dataset.subsets("ah")["test"]

    nose.tools.eq_(len(dataset["test"]), 10)
    for sample in dataset["test"]:
        nose.tools.eq_(len(sample), 3)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 608, 704)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 608, 704)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)


@stare_variable_set("bob.ip.binseg.stare.datadir")
def test_stare_vk():

    from ..configs.datasets.stare.vk import dataset
    # hack to allow testing on the CI
    dataset["train"]._samples = stare_dataset.subsets("vk")["train"]

    nose.tools.eq_(len(dataset["train"]), 10)
    for sample in dataset["train"]:
        nose.tools.eq_(len(sample), 3)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 608, 704)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 608, 704)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)

    # hack to allow testing on the CI
    dataset["test"]._samples = stare_dataset.subsets("vk")["test"]

    nose.tools.eq_(len(dataset["test"]), 10)
    for sample in dataset["test"]:
        nose.tools.eq_(len(sample), 3)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 608, 704)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 608, 704)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)


@rc_variable_set("bob.ip.binseg.chasedb1.datadir")
def test_chasedb1_first_annotator():

    from ..configs.datasets.chasedb1.first_annotator import dataset

    nose.tools.eq_(len(dataset["train"]), 8)
    for sample in dataset["train"]:
        nose.tools.eq_(len(sample), 3)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 960, 960)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 960, 960)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)

    nose.tools.eq_(len(dataset["test"]), 20)
    for sample in dataset["test"]:
        nose.tools.eq_(len(sample), 3)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 960, 960)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 960, 960)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)


@rc_variable_set("bob.ip.binseg.chasedb1.datadir")
def test_chasedb1_second_annotator():

    from ..configs.datasets.chasedb1.second_annotator import dataset

    nose.tools.eq_(len(dataset["train"]), 8)
    for sample in dataset["train"]:
        nose.tools.eq_(len(sample), 3)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 960, 960)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 960, 960)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)

    nose.tools.eq_(len(dataset["test"]), 20)
    for sample in dataset["test"]:
        nose.tools.eq_(len(sample), 3)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 960, 960)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 960, 960)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)


@rc_variable_set("bob.ip.binseg.hrf.datadir")
def test_hrf_default():

    from ..configs.datasets.hrf.default import dataset

    nose.tools.eq_(len(dataset["train"]), 15)
    for sample in dataset["train"]:
        nose.tools.eq_(len(sample), 4)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 1168, 1648)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 1168, 1648)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)
        nose.tools.eq_(sample[3].shape, (1, 1168, 1648)) #planes, height, width
        nose.tools.eq_(sample[3].dtype, torch.float32)

    nose.tools.eq_(len(dataset["test"]), 30)
    for sample in dataset["test"]:
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
def test_refuge_disc():

    from ..configs.datasets.refuge.disc import dataset

    nose.tools.eq_(len(dataset["train"]), 400)
    for sample in dataset["train"]:
        nose.tools.eq_(len(sample), 3)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 1632, 1632)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 1632, 1632)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)

    nose.tools.eq_(len(dataset["validation"]), 400)
    for sample in dataset["validation"]:
        nose.tools.eq_(len(sample), 3)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 1632, 1632)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 1632, 1632)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)

    nose.tools.eq_(len(dataset["test"]), 400)
    for sample in dataset["test"]:
        nose.tools.eq_(len(sample), 3)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 1632, 1632)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 1632, 1632)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)


@rc_variable_set("bob.ip.binseg.refuge.datadir")
@attr("slow")
def test_refuge_cup():

    from ..configs.datasets.refuge.cup import dataset

    nose.tools.eq_(len(dataset["train"]), 400)
    for sample in dataset["train"]:
        nose.tools.eq_(len(sample), 3)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 1632, 1632)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 1632, 1632)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)

    nose.tools.eq_(len(dataset["validation"]), 400)
    for sample in dataset["validation"]:
        nose.tools.eq_(len(sample), 3)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 1632, 1632)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 1632, 1632)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)

    nose.tools.eq_(len(dataset["test"]), 400)
    for sample in dataset["test"]:
        nose.tools.eq_(len(sample), 3)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 1632, 1632)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 1632, 1632)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)


@rc_variable_set("bob.ip.binseg.drishtigs1.datadir")
def test_drishtigs1_disc_all():

    from ..configs.datasets.drishtigs1.disc_all import dataset

    nose.tools.eq_(len(dataset["train"]), 50)
    for sample in dataset["train"]:
        nose.tools.eq_(len(sample), 3)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 1760, 2048)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 1760, 2048)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)

    nose.tools.eq_(len(dataset["test"]), 51)
    for sample in dataset["test"]:
        nose.tools.eq_(len(sample), 3)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 1760, 2048)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 1760, 2048)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)


@rc_variable_set("bob.ip.binseg.drishtigs1.datadir")
def test_drishtigs1_cup_all():

    from ..configs.datasets.drishtigs1.cup_all import dataset

    nose.tools.eq_(len(dataset["train"]), 50)
    for sample in dataset["train"]:
        nose.tools.eq_(len(sample), 3)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 1760, 2048)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 1760, 2048)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)

    nose.tools.eq_(len(dataset["test"]), 51)
    for sample in dataset["test"]:
        nose.tools.eq_(len(sample), 3)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 1760, 2048)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 1760, 2048)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)


@rc_variable_set("bob.ip.binseg.drionsdb.datadir")
def test_drionsdb_expert1():

    from ..configs.datasets.drionsdb.expert1 import dataset

    nose.tools.eq_(len(dataset["train"]), 60)
    for sample in dataset["train"]:
        nose.tools.eq_(len(sample), 3)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 416, 608)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 416, 608)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)

    nose.tools.eq_(len(dataset["test"]), 50)
    for sample in dataset["test"]:
        nose.tools.eq_(len(sample), 3)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 416, 608)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 416, 608)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)


@rc_variable_set("bob.ip.binseg.stare.datadir")
@rc_variable_set("bob.ip.binseg.chasedb1.datadir")
@rc_variable_set("bob.ip.binseg.hrf.datadir")
@rc_variable_set("bob.ip.binseg.iostar.datadir")
def test_drive_covd():

    from ..configs.datasets.drive.covd import dataset

    nose.tools.eq_(len(dataset["train"]), 53)
    for sample in dataset["train"]:
        assert 3 <= len(sample) <= 4
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 544, 544)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 544, 544)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)
        if len(sample) == 4:
            nose.tools.eq_(sample[3].shape, (1, 544, 544)) #planes, height, width
            nose.tools.eq_(sample[3].dtype, torch.float32)


@rc_variable_set("bob.ip.binseg.drive.datadir")
@rc_variable_set("bob.ip.binseg.stare.datadir")
@rc_variable_set("bob.ip.binseg.chasedb1.datadir")
@rc_variable_set("bob.ip.binseg.hrf.datadir")
@rc_variable_set("bob.ip.binseg.iostar.datadir")
def test_drive_ssl():

    from ..configs.datasets.drive.ssl import dataset

    nose.tools.eq_(len(dataset["train"]), 53)
    for sample in dataset["train"]:
        assert 5 <= len(sample) <= 6
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 544, 544)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 544, 544)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)
        if len(sample) == 6:
            nose.tools.eq_(sample[3].shape, (1, 544, 544)) #planes, height, width
            nose.tools.eq_(sample[3].dtype, torch.float32)
            assert isinstance(sample[4], str)
            nose.tools.eq_(sample[5].shape, (3, 544, 544)) #planes, height, width
            nose.tools.eq_(sample[5].dtype, torch.float32)
        else:
            assert isinstance(sample[3], str)
            nose.tools.eq_(sample[4].shape, (3, 544, 544)) #planes, height, width
            nose.tools.eq_(sample[4].dtype, torch.float32)


@rc_variable_set("bob.ip.binseg.drive.datadir")
@rc_variable_set("bob.ip.binseg.chasedb1.datadir")
@rc_variable_set("bob.ip.binseg.hrf.datadir")
@rc_variable_set("bob.ip.binseg.iostar.datadir")
def test_stare_covd():

    from ..configs.datasets.stare.covd import dataset

    nose.tools.eq_(len(dataset["train"]), 63)
    for sample in dataset["train"]:
        assert 3 <= len(sample) <= 4
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 608, 704)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 608, 704)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)
        if len(sample) == 4:
            nose.tools.eq_(sample[3].shape, (1, 608, 704)) #planes, height, width
            nose.tools.eq_(sample[3].dtype, torch.float32)


@rc_variable_set("bob.ip.binseg.drive.datadir")
@rc_variable_set("bob.ip.binseg.stare.datadir")
@rc_variable_set("bob.ip.binseg.hrf.datadir")
@rc_variable_set("bob.ip.binseg.iostar.datadir")
def test_chasedb1_covd():

    from ..configs.datasets.chasedb1.covd import dataset

    nose.tools.eq_(len(dataset["train"]), 65)
    for sample in dataset["train"]:
        assert 3 <= len(sample) <= 4
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 960, 960)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 960, 960)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)
        if len(sample) == 4:
            nose.tools.eq_(sample[3].shape, (1, 960, 960)) #planes, height, width
            nose.tools.eq_(sample[3].dtype, torch.float32)


@rc_variable_set("bob.ip.binseg.drive.datadir")
@rc_variable_set("bob.ip.binseg.stare.datadir")
@rc_variable_set("bob.ip.binseg.chasedb1.datadir")
@rc_variable_set("bob.ip.binseg.iostar.datadir")
def test_hrf_covd():

    from ..configs.datasets.hrf.covd import dataset

    nose.tools.eq_(len(dataset["train"]), 58)
    for sample in dataset["train"]:
        assert 3 <= len(sample) <= 4
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 1168, 1648)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 1168, 1648)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)
        if len(sample) == 4:
            nose.tools.eq_(sample[3].shape, (1, 1168, 1648))
            nose.tools.eq_(sample[3].dtype, torch.float32)


@rc_variable_set("bob.ip.binseg.drive.datadir")
@rc_variable_set("bob.ip.binseg.stare.datadir")
@rc_variable_set("bob.ip.binseg.chasedb1.datadir")
@rc_variable_set("bob.ip.binseg.hrf.datadir")
def test_iostar_covd():

    from ..configs.datasets.iostar.covd import dataset

    nose.tools.eq_(len(dataset["train"]), 53)
    for sample in dataset["train"]:
        assert 3 <= len(sample) <= 4
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 1024, 1024)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 1024, 1024)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)
        if len(sample) == 4:
            nose.tools.eq_(sample[3].shape, (1, 1024, 1024))
            nose.tools.eq_(sample[3].dtype, torch.float32)
