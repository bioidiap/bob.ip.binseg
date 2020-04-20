#!/usr/bin/env python
# coding=utf-8

import nose.tools

import torch

from . import mock_dataset
stare_dataset, stare_variable_set = mock_dataset()
from .utils import rc_variable_set

# we only iterate over the first N elements at most - dataset loading has
# already been checked on the individual datset tests.  Here, we are only
# testing for the extra tools wrapping the dataset
N = 10


@rc_variable_set("bob.ip.binseg.drive.datadir")
def test_drive_default():

    from ..configs.datasets.drive.default import dataset
    nose.tools.eq_(len(dataset["train"]), 20)
    nose.tools.eq_(dataset["train"].augmented, True)
    for sample in dataset["train"][:N]:
        nose.tools.eq_(len(sample), 4)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 544, 544)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 544, 544)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)
        nose.tools.eq_(sample[3].shape, (1, 544, 544)) #planes, height, width
        nose.tools.eq_(sample[3].dtype, torch.float32)

    nose.tools.eq_(len(dataset["test"]), 20)
    nose.tools.eq_(dataset["test"].augmented, False)
    for sample in dataset["test"][:N]:
        nose.tools.eq_(len(sample), 4)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 544, 544)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 544, 544)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)
        nose.tools.eq_(sample[3].shape, (1, 544, 544)) #planes, height, width
        nose.tools.eq_(sample[3].dtype, torch.float32)


@stare_variable_set("bob.ip.binseg.stare.datadir")
def test_stare_augmentation_manipulation():

    # some tests to check our context management for dataset augmentation works
    # adequately, with one example dataset

    from ..configs.datasets.stare.ah import dataset
    # hack to allow testing on the CI
    dataset["train"]._samples = stare_dataset.subsets("ah")["train"]

    nose.tools.eq_(dataset["train"].augmented, True)
    nose.tools.eq_(dataset["test"].augmented, False)
    nose.tools.eq_(len(dataset["train"]._transforms.transforms),
            len(dataset["test"]._transforms.transforms) + 4)

    with dataset["train"].not_augmented() as d:
        nose.tools.eq_(len(d._transforms.transforms), 2)
        nose.tools.eq_(d.augmented, False)
        nose.tools.eq_(dataset["train"].augmented, False)
        nose.tools.eq_(dataset["test"].augmented, False)

    nose.tools.eq_(dataset["train"].augmented, True)
    nose.tools.eq_(dataset["test"].augmented, False)
    nose.tools.eq_(len(dataset["train"]._transforms.transforms),
            len(dataset["test"]._transforms.transforms) + 4)


@stare_variable_set("bob.ip.binseg.stare.datadir")
def test_stare_ah():

    from ..configs.datasets.stare.ah import dataset
    # hack to allow testing on the CI
    dataset["train"]._samples = stare_dataset.subsets("ah")["train"]

    nose.tools.eq_(len(dataset["train"]), 10)
    nose.tools.eq_(dataset["train"].augmented, True)
    for sample in dataset["train"][:N]:
        nose.tools.eq_(len(sample), 3)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 608, 704)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 608, 704)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)

    # hack to allow testing on the CI
    dataset["test"]._samples = stare_dataset.subsets("ah")["test"]

    nose.tools.eq_(len(dataset["test"]), 10)
    nose.tools.eq_(dataset["test"].augmented, False)
    for sample in dataset["test"][:N]:
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
    nose.tools.eq_(dataset["train"].augmented, True)
    for sample in dataset["train"][:N]:
        nose.tools.eq_(len(sample), 3)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 608, 704)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 608, 704)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)

    # hack to allow testing on the CI
    dataset["test"]._samples = stare_dataset.subsets("vk")["test"]

    nose.tools.eq_(len(dataset["test"]), 10)
    nose.tools.eq_(dataset["test"].augmented, False)
    for sample in dataset["test"][:N]:
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
    nose.tools.eq_(dataset["train"].augmented, True)
    for sample in dataset["train"][:N]:
        nose.tools.eq_(len(sample), 3)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 960, 960)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 960, 960)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)

    nose.tools.eq_(len(dataset["test"]), 20)
    nose.tools.eq_(dataset["test"].augmented, False)
    for sample in dataset["test"][:N]:
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
    nose.tools.eq_(dataset["train"].augmented, True)
    for sample in dataset["train"][:N]:
        nose.tools.eq_(len(sample), 3)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 960, 960)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 960, 960)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)

    nose.tools.eq_(len(dataset["test"]), 20)
    nose.tools.eq_(dataset["test"].augmented, False)
    for sample in dataset["test"][:N]:
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
    nose.tools.eq_(dataset["train"].augmented, True)
    for sample in dataset["train"][:N]:
        nose.tools.eq_(len(sample), 4)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 1168, 1648)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 1168, 1648)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)
        nose.tools.eq_(sample[3].shape, (1, 1168, 1648)) #planes, height, width
        nose.tools.eq_(sample[3].dtype, torch.float32)

    nose.tools.eq_(len(dataset["test"]), 30)
    nose.tools.eq_(dataset["test"].augmented, False)
    for sample in dataset["test"][:N]:
        nose.tools.eq_(len(sample), 4)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 1168, 1648)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 1168, 1648)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)
        nose.tools.eq_(sample[3].shape, (1, 1168, 1648)) #planes, height, width
        nose.tools.eq_(sample[3].dtype, torch.float32)


@rc_variable_set("bob.ip.binseg.refuge.datadir")
def test_refuge_disc():

    from ..configs.datasets.refuge.disc import dataset

    nose.tools.eq_(len(dataset["train"]), 400)
    nose.tools.eq_(dataset["train"].augmented, True)
    for sample in dataset["train"][:N]:
        nose.tools.eq_(len(sample), 3)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 1632, 1632)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 1632, 1632)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)

    nose.tools.eq_(len(dataset["validation"]), 400)
    nose.tools.eq_(dataset["validation"].augmented, False)
    for sample in dataset["validation"][:N]:
        nose.tools.eq_(len(sample), 3)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 1632, 1632)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 1632, 1632)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)

    nose.tools.eq_(len(dataset["test"]), 400)
    nose.tools.eq_(dataset["test"].augmented, False)
    for sample in dataset["test"][:N]:
        nose.tools.eq_(len(sample), 3)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 1632, 1632)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 1632, 1632)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)


@rc_variable_set("bob.ip.binseg.refuge.datadir")
def test_refuge_cup():

    from ..configs.datasets.refuge.cup import dataset

    nose.tools.eq_(len(dataset["train"]), 400)
    nose.tools.eq_(dataset["train"].augmented, True)
    for sample in dataset["train"][:N]:
        nose.tools.eq_(len(sample), 3)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 1632, 1632)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 1632, 1632)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)

    nose.tools.eq_(len(dataset["validation"]), 400)
    nose.tools.eq_(dataset["validation"].augmented, False)
    for sample in dataset["validation"][:N]:
        nose.tools.eq_(len(sample), 3)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 1632, 1632)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 1632, 1632)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)

    nose.tools.eq_(len(dataset["test"]), 400)
    nose.tools.eq_(dataset["test"].augmented, False)
    for sample in dataset["test"][:N]:
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
    nose.tools.eq_(dataset["train"].augmented, True)
    for sample in dataset["train"][:N]:
        nose.tools.eq_(len(sample), 3)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 1760, 2048)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 1760, 2048)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)

    nose.tools.eq_(len(dataset["test"]), 51)
    nose.tools.eq_(dataset["test"].augmented, False)
    for sample in dataset["test"][:N]:
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
    nose.tools.eq_(dataset["train"].augmented, True)
    for sample in dataset["train"][:N]:
        nose.tools.eq_(len(sample), 3)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 1760, 2048)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 1760, 2048)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)

    nose.tools.eq_(len(dataset["test"]), 51)
    nose.tools.eq_(dataset["test"].augmented, False)
    for sample in dataset["test"][:N]:
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
    nose.tools.eq_(dataset["train"].augmented, True)
    for sample in dataset["train"][:N]:
        nose.tools.eq_(len(sample), 3)
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 416, 608)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 416, 608)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)

    nose.tools.eq_(len(dataset["test"]), 50)
    nose.tools.eq_(dataset["test"].augmented, False)
    for sample in dataset["test"][:N]:
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
    #nose.tools.eq_(dataset["train"].augmented, True)  ##ConcatDataset
    nose.tools.eq_(dataset["test"].augmented, False)
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
    #nose.tools.eq_(dataset["train"].augmented, True)  ##ConcatDataset
    nose.tools.eq_(dataset["test"].augmented, False)
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
    #nose.tools.eq_(dataset["train"].augmented, True)  ##ConcatDataset
    nose.tools.eq_(dataset["test"].augmented, False)
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
    #nose.tools.eq_(dataset["train"].augmented, True)  ##ConcatDataset
    nose.tools.eq_(dataset["test"].augmented, False)
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
    #nose.tools.eq_(dataset["train"].augmented, True)  ##ConcatDataset
    nose.tools.eq_(dataset["test"].augmented, False)
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
    #nose.tools.eq_(dataset["train"].augmented, True)  ##ConcatDataset
    nose.tools.eq_(dataset["test"].augmented, False)
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
