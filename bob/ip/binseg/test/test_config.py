#!/usr/bin/env python
# coding=utf-8

import importlib
import torch
import pytest

from . import mock_dataset
stare_datadir, stare_dataset = mock_dataset()

# we only iterate over the first N elements at most - dataset loading has
# already been checked on the individual datset tests.  Here, we are only
# testing for the extra tools wrapping the dataset
N = 10


@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.drive.datadir")
def test_drive():

    def _check_subset(samples, size):
        assert len(samples) == size
        for s in samples:
            assert len(s) == 4
            assert isinstance(s[0], str)
            assert s[1].shape, (3, 544 == 544) #planes, height, width
            assert s[1].dtype == torch.float32
            assert s[2].shape, (1, 544 == 544) #planes, height, width
            assert s[2].dtype == torch.float32
            assert s[3].shape, (1, 544 == 544) #planes, height, width
            assert s[3].dtype == torch.float32
            assert s[1].max() <= 1.0
            assert s[1].min() >= 0.0

    from ..configs.datasets.drive.default import dataset

    assert len(dataset) == 4
    _check_subset(dataset["__train__"], 20)
    _check_subset(dataset["__valid__"], 20)
    _check_subset(dataset["train"], 20)
    _check_subset(dataset["test"], 20)

    from ..configs.datasets.drive.second_annotator import dataset

    assert len(dataset) == 1
    _check_subset(dataset["test"], 20)


@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.drive.datadir")
@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.chasedb1.datadir")
@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.hrf.datadir")
@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.iostar.datadir")
def test_drive_mtest():

    from ..configs.datasets.drive.mtest import dataset
    assert len(dataset) == 10

    from ..configs.datasets.drive.default import dataset as baseline
    assert dataset["train"] == baseline["train"]
    assert dataset["test"] == baseline["test"]

    for subset in dataset:
        for sample in dataset[subset]:
            assert len(sample) == 4
            assert isinstance(sample[0], str)
            assert sample[1].shape, (3, 544 == 544) #planes, height, width
            assert sample[1].dtype == torch.float32
            assert sample[2].shape, (1, 544 == 544)
            assert sample[2].dtype == torch.float32
            assert sample[3].shape, (1, 544 == 544)
            assert sample[3].dtype == torch.float32
            assert sample[1].max() <= 1.0
            assert sample[1].min() >= 0.0


@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.drive.datadir")
@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.chasedb1.datadir")
@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.hrf.datadir")
@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.iostar.datadir")
def test_drive_covd():

    from ..configs.datasets.drive.covd import dataset
    assert len(dataset) == 4

    from ..configs.datasets.drive.default import dataset as baseline
    assert dataset["train"] == dataset["__valid__"]
    assert dataset["test"] == baseline["test"]

    for key in ("__train__", "train"):
        assert len(dataset[key]) == 123
        for sample in dataset["__train__"]:
            assert len(sample) == 4
            assert isinstance(sample[0], str)
            assert sample[1].shape, (3, 544 == 544) #planes, height, width
            assert sample[1].dtype == torch.float32
            assert sample[2].shape, (1, 544 == 544) #planes, height, width
            assert sample[2].dtype == torch.float32
            assert sample[3].shape, (1, 544 == 544)
            assert sample[3].dtype == torch.float32
            assert sample[1].max() <= 1.0
            assert sample[1].min() >= 0.0


@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.drive.datadir")
@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.chasedb1.datadir")
@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.hrf.datadir")
@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.iostar.datadir")
def test_drive_ssl():

    from ..configs.datasets.drive.ssl import dataset
    assert len(dataset) == 4

    from ..configs.datasets.drive.covd import dataset as covd
    assert dataset["train"] == covd["train"]
    assert dataset["train"] == dataset["__valid__"]
    assert dataset["test"] == covd["test"]
    assert dataset["__valid__"] == covd["__valid__"]

    # these are the only different from the baseline
    assert len(dataset["__train__"]) == 123
    for sample in dataset["__train__"]:
        assert len(sample) == 6
        assert isinstance(sample[0], str)
        assert sample[1].shape, (3, 544 == 544) #planes, height, width
        assert sample[1].dtype == torch.float32
        assert sample[2].shape, (1, 544 == 544) #planes, height, width
        assert sample[2].dtype == torch.float32
        assert sample[3].shape, (1, 544 == 544) #planes, height, width
        assert sample[3].dtype == torch.float32
        assert isinstance(sample[4], str)
        assert sample[5].shape, (3, 544 == 544) #planes, height, width
        assert sample[5].dtype == torch.float32
        assert sample[1].max() <= 1.0
        assert sample[1].min() >= 0.0


def test_stare_augmentation_manipulation():

    # some tests to check our context management for dataset augmentation works
    # adequately, with one example dataset

    # hack to allow testing on the CI
    from ..configs.datasets.stare import _maker
    dataset = _maker("ah", stare_dataset)

    assert len(dataset["__train__"]._transforms.transforms) == \
            (len(dataset["test"]._transforms.transforms) + 4)

    assert len(dataset["train"]._transforms.transforms) == \
            len(dataset["test"]._transforms.transforms)


def test_stare():

    def _check_subset(samples, size):
        assert len(samples) == size
        for s in samples:
            assert len(s) == 4
            assert isinstance(s[0], str)
            assert s[1].shape, (3, 608 == 704) #planes, height, width
            assert s[1].dtype == torch.float32
            assert s[2].shape, (1, 608 == 704) #planes, height, width
            assert s[2].dtype == torch.float32
            assert s[3].shape, (1, 608 == 704) #planes, height, width
            assert s[3].dtype == torch.float32
            assert s[1].max() <= 1.0
            assert s[1].min() >= 0.0

    # hack to allow testing on the CI
    from ..configs.datasets.stare import _maker

    for protocol in "ah", "vk":
        dataset = _maker(protocol, stare_dataset)
        assert len(dataset) == 4
        _check_subset(dataset["__train__"], 10)
        _check_subset(dataset["train"], 10)
        _check_subset(dataset["test"], 10)


@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.drive.datadir")
@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.chasedb1.datadir")
@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.hrf.datadir")
@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.iostar.datadir")
def test_stare_mtest():

    from ..configs.datasets.stare.mtest import dataset
    assert len(dataset) == 10

    from ..configs.datasets.stare.ah import dataset as baseline
    assert dataset["train"] == baseline["train"]
    assert dataset["test"] == baseline["test"]

    for subset in dataset:
        for sample in dataset[subset]:
            assert len(sample) == 4
            assert isinstance(sample[0], str)
            assert sample[1].shape, (3, 608 == 704) #planes,height,width
            assert sample[1].dtype == torch.float32
            assert sample[2].shape, (1, 608 == 704) #planes,height,width
            assert sample[2].dtype == torch.float32
            assert sample[3].shape, (1, 608 == 704)
            assert sample[3].dtype == torch.float32
            assert sample[1].max() <= 1.0
            assert sample[1].min() >= 0.0


@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.drive.datadir")
@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.chasedb1.datadir")
@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.hrf.datadir")
@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.iostar.datadir")
def test_stare_covd():

    from ..configs.datasets.stare.covd import dataset
    assert len(dataset) == 4

    from ..configs.datasets.stare.ah import dataset as baseline
    assert dataset["train"] == dataset["__valid__"]
    assert dataset["test"] == baseline["test"]

    # these are the only different sets from the baseline
    for key in ("__train__", "train"):
        assert len(dataset[key]) == 143
        for sample in dataset[key]:
            assert len(sample) == 4
            assert isinstance(sample[0], str)
            assert sample[1].shape, (3, 608 == 704) #planes, height, width
            assert sample[1].dtype == torch.float32
            assert sample[2].shape, (1, 608 == 704) #planes, height, width
            assert sample[2].dtype == torch.float32
            assert sample[1].max() <= 1.0
            assert sample[1].min() >= 0.0
            assert sample[3].shape, (1, 608 == 704)
            assert sample[3].dtype == torch.float32


@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.chasedb1.datadir")
def test_chasedb1():

    def _check_subset(samples, size):
        assert len(samples) == size
        for s in samples:
            assert len(s) == 4
            assert isinstance(s[0], str)
            assert s[1].shape, (3, 960 == 960) #planes, height, width
            assert s[1].dtype == torch.float32
            assert s[2].shape, (1, 960 == 960) #planes, height, width
            assert s[2].dtype == torch.float32
            assert s[3].shape, (1, 960 == 960) #planes, height, width
            assert s[3].dtype == torch.float32
            assert s[1].max() <= 1.0
            assert s[1].min() >= 0.0

    for m in ("first_annotator", "second_annotator"):
        d = importlib.import_module(f"...configs.datasets.chasedb1.{m}",
                package=__name__).dataset
        assert len(d) == 4
        _check_subset(d["__train__"], 8)
        _check_subset(d["__valid__"], 8)
        _check_subset(d["train"], 8)
        _check_subset(d["test"], 20)


@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.drive.datadir")
@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.chasedb1.datadir")
@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.hrf.datadir")
@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.iostar.datadir")
def test_chasedb1_mtest():

    from ..configs.datasets.chasedb1.mtest import dataset
    assert len(dataset) == 10

    from ..configs.datasets.chasedb1.first_annotator import dataset as baseline
    assert dataset["train"] == baseline["train"]
    assert dataset["test"] == baseline["test"]

    for subset in dataset:
        for sample in dataset[subset]:
            assert len(sample) == 4
            assert isinstance(sample[0], str)
            assert sample[1].shape, (3, 960 == 960) #planes,height,width
            assert sample[1].dtype == torch.float32
            assert sample[2].shape, (1, 960 == 960) #planes,height,width
            assert sample[2].dtype == torch.float32
            assert sample[3].shape, (1, 960 == 960)
            assert sample[3].dtype == torch.float32
            assert sample[1].max() <= 1.0
            assert sample[1].min() >= 0.0


@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.drive.datadir")
@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.chasedb1.datadir")
@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.hrf.datadir")
@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.iostar.datadir")
def test_chasedb1_covd():

    from ..configs.datasets.chasedb1.covd import dataset
    assert len(dataset) == 4

    from ..configs.datasets.chasedb1.first_annotator import dataset as baseline
    assert dataset["train"] == dataset["__valid__"]
    assert dataset["test"] == baseline["test"]

    # these are the only different sets from the baseline
    for key in ("__train__", "train"):
        assert len(dataset[key]) == 135
        for sample in dataset[key]:
            assert len(sample) == 4
            assert isinstance(sample[0], str)
            assert sample[1].shape, (3, 960 == 960) #planes, height, width
            assert sample[1].dtype == torch.float32
            assert sample[2].shape, (1, 960 == 960) #planes, height, width
            assert sample[2].dtype == torch.float32
            assert sample[3].shape, (1, 960 == 960)
            assert sample[3].dtype == torch.float32
            assert sample[1].max() <= 1.0
            assert sample[1].min() >= 0.0


@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.hrf.datadir")
def test_hrf():

    def _check_subset(samples, size):
        assert len(samples) == size
        for s in samples:
            assert len(s) == 4
            assert isinstance(s[0], str)
            assert s[1].shape, (3, 1168 == 1648) #planes, height, width
            assert s[1].dtype == torch.float32
            assert s[2].shape, (1, 1168 == 1648) #planes, height, width
            assert s[2].dtype == torch.float32
            assert s[3].shape, (1, 1168 == 1648) #planes, height, width
            assert s[3].dtype == torch.float32
            assert s[1].max() <= 1.0
            assert s[1].min() >= 0.0

    def _check_subset_fullres(samples, size):
        assert len(samples) == size
        for s in samples:
            assert len(s) == 4
            assert isinstance(s[0], str)
            assert s[1].shape, (3, 2336 == 3296) #planes, height, width
            assert s[1].dtype == torch.float32
            assert s[2].shape, (1, 2336 == 3296) #planes, height, width
            assert s[2].dtype == torch.float32
            assert s[3].shape, (1, 2336 == 3296) #planes, height, width
            assert s[3].dtype == torch.float32
            assert s[1].max() <= 1.0
            assert s[1].min() >= 0.0

    from ..configs.datasets.hrf.default import dataset
    assert len(dataset) == 6
    _check_subset(dataset["__train__"], 15)
    _check_subset(dataset["train"], 15)
    _check_subset(dataset["test"], 30)
    _check_subset_fullres(dataset["train (full resolution)"], 15)
    _check_subset_fullres(dataset["test (full resolution)"], 30)


@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.drive.datadir")
@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.chasedb1.datadir")
@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.hrf.datadir")
@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.iostar.datadir")
def test_hrf_mtest():

    from ..configs.datasets.hrf.mtest import dataset
    assert len(dataset) == 12

    from ..configs.datasets.hrf.default import dataset as baseline
    assert dataset["train"] == baseline["train"]
    assert dataset["test"] == baseline["test"]

    for subset in dataset:
        for sample in dataset[subset]:
            assert len(sample) == 4
            assert isinstance(sample[0], str)
            if "full resolution" in subset:
                assert sample[1].shape, (3, 2336 == 3296)
                assert sample[1].dtype == torch.float32
                assert sample[2].shape, (1, 2336 == 3296)
                assert sample[2].dtype == torch.float32
                assert sample[3].shape, (1, 2336 == 3296)
                assert sample[3].dtype == torch.float32
            else:
                assert sample[1].shape, (3, 1168 == 1648)
                assert sample[1].dtype == torch.float32
                assert sample[2].shape, (1, 1168 == 1648)
                assert sample[2].dtype == torch.float32
                assert sample[3].shape, (1, 1168 == 1648)
                assert sample[3].dtype == torch.float32
            assert sample[1].max() <= 1.0
            assert sample[1].min() >= 0.0


@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.drive.datadir")
@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.chasedb1.datadir")
@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.hrf.datadir")
@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.iostar.datadir")
def test_hrf_covd():

    from ..configs.datasets.hrf.covd import dataset
    assert len(dataset) == 6

    from ..configs.datasets.hrf.default import dataset as baseline
    assert dataset["train"] == dataset["__valid__"]
    assert dataset["test"] == baseline["test"]

    # these are the only different sets from the baseline
    for key in ("__train__", "train"):
        assert len(dataset[key]) == 118
        for sample in dataset[key]:
            assert len(sample) == 4
            assert isinstance(sample[0], str)
            assert sample[1].shape, (3, 1168 == 1648)
            assert sample[1].dtype == torch.float32
            assert sample[2].shape, (1, 1168 == 1648)
            assert sample[2].dtype == torch.float32
            assert sample[3].shape, (1, 1168 == 1648)
            assert sample[3].dtype == torch.float32
            assert sample[1].max() <= 1.0
            assert sample[1].min() >= 0.0


@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.iostar.datadir")
def test_iostar():

    def _check_subset(samples, size):
        assert len(samples) == size
        for s in samples:
            assert len(s) == 4
            assert isinstance(s[0], str)
            assert s[1].shape, (3, 1024 == 1024) #planes, height, width
            assert s[1].dtype == torch.float32
            assert s[2].shape, (1, 1024 == 1024) #planes, height, width
            assert s[2].dtype == torch.float32
            assert s[3].shape, (1, 1024 == 1024) #planes, height, width
            assert s[3].dtype == torch.float32
            assert s[1].max() <= 1.0
            assert s[1].min() >= 0.0

    for m in ("vessel", "optic_disc"):
        d = importlib.import_module(f"...configs.datasets.iostar.{m}",
                package=__name__).dataset
        assert len(d) == 4
        _check_subset(d["__train__"], 20)
        _check_subset(d["train"], 20)
        _check_subset(d["test"], 10)


@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.drive.datadir")
@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.chasedb1.datadir")
@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.hrf.datadir")
@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.iostar.datadir")
def test_iostar_mtest():

    from ..configs.datasets.iostar.vessel_mtest import dataset
    assert len(dataset) == 10

    from ..configs.datasets.iostar.vessel import dataset as baseline
    assert dataset["train"] == baseline["train"]
    assert dataset["test"] == baseline["test"]

    for subset in dataset:
        for sample in dataset[subset]:
            assert len(sample) == 4
            assert isinstance(sample[0], str)
            assert sample[1].shape, (3, 1024 == 1024) #planes,height,width
            assert sample[1].dtype == torch.float32
            assert sample[2].shape, (1, 1024 == 1024) #planes,height,width
            assert sample[2].dtype == torch.float32
            assert sample[3].shape, (1, 1024 == 1024)
            assert sample[3].dtype == torch.float32
            assert sample[1].max() <= 1.0
            assert sample[1].min() >= 0.0


@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.drive.datadir")
@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.chasedb1.datadir")
@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.hrf.datadir")
@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.iostar.datadir")
def test_iostar_covd():

    from ..configs.datasets.iostar.covd import dataset
    assert len(dataset) == 4

    from ..configs.datasets.iostar.vessel import dataset as baseline
    assert dataset["train"] == dataset["__valid__"]
    assert dataset["test"] == baseline["test"]

    # these are the only different sets from the baseline
    for key in ("__train__", "train"):
        assert len(dataset[key]) == 133
        for sample in dataset[key]:
            assert len(sample) == 4
            assert isinstance(sample[0], str)
            assert sample[1].shape, (3, 1024 == 1024)
            assert sample[1].dtype == torch.float32
            assert sample[2].shape, (1, 1024 == 1024)
            assert sample[2].dtype == torch.float32
            assert sample[3].shape, (1, 1024 == 1024)
            assert sample[3].dtype == torch.float32
            assert sample[1].max() <= 1.0
            assert sample[1].min() >= 0.0


@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.refuge.datadir")
def test_refuge():

    def _check_subset(samples, size):
        assert len(samples) == size
        for s in samples[:N]:
            assert len(s) == 3
            assert isinstance(s[0], str)
            assert s[1].shape, (3, 1632 == 1632) #planes, height, width
            assert s[1].dtype == torch.float32
            assert s[2].shape, (1, 1632 == 1632) #planes, height, width
            assert s[2].dtype == torch.float32
            assert s[1].max() <= 1.0
            assert s[1].min() >= 0.0

    for m in ("disc", "cup"):
        d = importlib.import_module(f"...configs.datasets.refuge.{m}",
                package=__name__).dataset
        assert len(d) == 5
        _check_subset(d["__train__"], 400)
        _check_subset(d["train"], 400)
        _check_subset(d["validation"], 400)
        _check_subset(d["test"], 400)


@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.drishtigs1.datadir")
def test_drishtigs1():

    def _check_subset(samples, size):
        assert len(samples) == size
        for s in samples[:N]:
            assert len(s) == 3
            assert isinstance(s[0], str)
            assert s[1].shape, (3, 1760 == 2048) #planes, height, width
            assert s[1].dtype == torch.float32
            assert s[2].shape, (1, 1760 == 2048) #planes, height, width
            assert s[2].dtype == torch.float32
            assert s[1].max() <= 1.0
            assert s[1].min() >= 0.0

    for m in ("disc_all", "cup_all", "disc_any", "cup_any"):
        d = importlib.import_module(f"...configs.datasets.drishtigs1.{m}",
                package=__name__).dataset
        assert len(d) == 4
        _check_subset(d["__train__"], 50)
        _check_subset(d["train"], 50)
        _check_subset(d["test"], 51)


@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.rimoner3.datadir")
def test_rimoner3():

    def _check_subset(samples, size):
        assert len(samples) == size
        for s in samples[:N]:
            assert len(s) == 3
            assert isinstance(s[0], str)
            assert s[1].shape, (3, 1440 == 1088) #planes, height, width
            assert s[1].dtype == torch.float32
            assert s[2].shape, (1, 1440 == 1088) #planes, height, width
            assert s[2].dtype == torch.float32
            assert s[1].max() <= 1.0
            assert s[1].min() >= 0.0

    for m in ("disc_exp1", "cup_exp1", "disc_exp2", "cup_exp2"):
        d = importlib.import_module(f"...configs.datasets.rimoner3.{m}",
                package=__name__).dataset
        assert len(d) == 4
        _check_subset(d["__train__"], 99)
        _check_subset(d["train"], 99)
        _check_subset(d["test"], 60)


@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.drionsdb.datadir")
def test_drionsdb():

    def _check_subset(samples, size):
        assert len(samples) == size
        for s in samples[:N]:
            assert len(s) == 3
            assert isinstance(s[0], str)
            assert s[1].shape, (3, 416 == 608) #planes, height, width
            assert s[1].dtype == torch.float32
            assert s[2].shape, (1, 416 == 608) #planes, height, width
            assert s[2].dtype == torch.float32
            assert s[1].max() <= 1.0
            assert s[1].min() >= 0.0

    for m in ("expert1", "expert2"):
        d = importlib.import_module(f"...configs.datasets.drionsdb.{m}",
                package=__name__).dataset
        assert len(d) == 4
        _check_subset(d["__train__"], 60)
        _check_subset(d["train"], 60)
        _check_subset(d["test"], 50)
