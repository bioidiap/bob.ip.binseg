#!/usr/bin/env python
# coding=utf-8

import importlib

import nose.tools

import torch

from . import mock_dataset
stare_datadir, stare_dataset, stare_variable_set = mock_dataset()
from .utils import rc_variable_set

# we only iterate over the first N elements at most - dataset loading has
# already been checked on the individual datset tests.  Here, we are only
# testing for the extra tools wrapping the dataset
N = 10


@rc_variable_set("bob.ip.binseg.drive.datadir")
def test_drive():

    def _check_subset(samples, size):
        nose.tools.eq_(len(samples), size)
        for s in samples:
            nose.tools.eq_(len(s), 4)
            assert isinstance(s[0], str)
            nose.tools.eq_(s[1].shape, (3, 544, 544)) #planes, height, width
            nose.tools.eq_(s[1].dtype, torch.float32)
            nose.tools.eq_(s[2].shape, (1, 544, 544)) #planes, height, width
            nose.tools.eq_(s[2].dtype, torch.float32)
            nose.tools.eq_(s[3].shape, (1, 544, 544)) #planes, height, width
            nose.tools.eq_(s[3].dtype, torch.float32)
            assert s[1].max() <= 1.0
            assert s[1].min() >= 0.0

    from ..configs.datasets.drive.default import dataset

    nose.tools.eq_(len(dataset), 4)
    _check_subset(dataset["__train__"], 20)
    _check_subset(dataset["__valid__"], 20)
    _check_subset(dataset["train"], 20)
    _check_subset(dataset["test"], 20)

    from ..configs.datasets.drive.second_annotator import dataset

    nose.tools.eq_(len(dataset), 1)
    _check_subset(dataset["test"], 20)


@rc_variable_set("bob.ip.binseg.drive.datadir")
@stare_variable_set("bob.ip.binseg.stare.datadir")
@rc_variable_set("bob.ip.binseg.chasedb1.datadir")
@rc_variable_set("bob.ip.binseg.hrf.datadir")
@rc_variable_set("bob.ip.binseg.iostar.datadir")
def test_drive_mtest():

    from ..configs.datasets.drive.mtest import dataset
    nose.tools.eq_(len(dataset), 10)

    from ..configs.datasets.drive.default import dataset as baseline
    nose.tools.eq_(dataset["train"], baseline["train"])
    nose.tools.eq_(dataset["test"], baseline["test"])

    for subset in dataset:
        for sample in dataset[subset]:
            assert len(sample) == 4
            assert isinstance(sample[0], str)
            nose.tools.eq_(sample[1].shape, (3, 544, 544)) #planes, height, width
            nose.tools.eq_(sample[1].dtype, torch.float32)
            nose.tools.eq_(sample[2].shape, (1, 544, 544))
            nose.tools.eq_(sample[2].dtype, torch.float32)
            nose.tools.eq_(sample[3].shape, (1, 544, 544))
            nose.tools.eq_(sample[3].dtype, torch.float32)
            assert sample[1].max() <= 1.0
            assert sample[1].min() >= 0.0


@rc_variable_set("bob.ip.binseg.drive.datadir")
@stare_variable_set("bob.ip.binseg.stare.datadir")
@rc_variable_set("bob.ip.binseg.chasedb1.datadir")
@rc_variable_set("bob.ip.binseg.hrf.datadir")
@rc_variable_set("bob.ip.binseg.iostar.datadir")
def test_drive_covd():

    from ..configs.datasets.drive.covd import dataset
    nose.tools.eq_(len(dataset), 4)

    from ..configs.datasets.drive.default import dataset as baseline
    nose.tools.eq_(dataset["train"], dataset["__valid__"])
    nose.tools.eq_(dataset["test"], baseline["test"])

    for key in ("__train__", "train"):
        nose.tools.eq_(len(dataset[key]), 123)
        for sample in dataset["__train__"]:
            assert len(sample) == 4
            assert isinstance(sample[0], str)
            nose.tools.eq_(sample[1].shape, (3, 544, 544)) #planes, height, width
            nose.tools.eq_(sample[1].dtype, torch.float32)
            nose.tools.eq_(sample[2].shape, (1, 544, 544)) #planes, height, width
            nose.tools.eq_(sample[2].dtype, torch.float32)
            nose.tools.eq_(sample[3].shape, (1, 544, 544))
            nose.tools.eq_(sample[3].dtype, torch.float32)
            assert sample[1].max() <= 1.0
            assert sample[1].min() >= 0.0


@rc_variable_set("bob.ip.binseg.drive.datadir")
@stare_variable_set("bob.ip.binseg.stare.datadir")
@rc_variable_set("bob.ip.binseg.chasedb1.datadir")
@rc_variable_set("bob.ip.binseg.hrf.datadir")
@rc_variable_set("bob.ip.binseg.iostar.datadir")
def test_drive_ssl():

    from ..configs.datasets.drive.ssl import dataset
    nose.tools.eq_(len(dataset), 4)

    from ..configs.datasets.drive.covd import dataset as covd
    nose.tools.eq_(dataset["train"], covd["train"])
    nose.tools.eq_(dataset["train"], dataset["__valid__"])
    nose.tools.eq_(dataset["test"], covd["test"])
    nose.tools.eq_(dataset["__valid__"], covd["__valid__"])

    # these are the only different from the baseline
    nose.tools.eq_(len(dataset["__train__"]), 123)
    for sample in dataset["__train__"]:
        assert len(sample) == 6
        assert isinstance(sample[0], str)
        nose.tools.eq_(sample[1].shape, (3, 544, 544)) #planes, height, width
        nose.tools.eq_(sample[1].dtype, torch.float32)
        nose.tools.eq_(sample[2].shape, (1, 544, 544)) #planes, height, width
        nose.tools.eq_(sample[2].dtype, torch.float32)
        nose.tools.eq_(sample[3].shape, (1, 544, 544)) #planes, height, width
        nose.tools.eq_(sample[3].dtype, torch.float32)
        assert isinstance(sample[4], str)
        nose.tools.eq_(sample[5].shape, (3, 544, 544)) #planes, height, width
        nose.tools.eq_(sample[5].dtype, torch.float32)
        assert sample[1].max() <= 1.0
        assert sample[1].min() >= 0.0


@stare_variable_set("bob.ip.binseg.stare.datadir")
def test_stare_augmentation_manipulation():

    # some tests to check our context management for dataset augmentation works
    # adequately, with one example dataset

    # hack to allow testing on the CI
    from ..configs.datasets.stare import _maker
    dataset = _maker("ah", stare_dataset)

    nose.tools.eq_(len(dataset["__train__"]._transforms.transforms),
            len(dataset["test"]._transforms.transforms) + 4)

    nose.tools.eq_(len(dataset["train"]._transforms.transforms),
            len(dataset["test"]._transforms.transforms))


@stare_variable_set("bob.ip.binseg.stare.datadir")
def test_stare():

    def _check_subset(samples, size):
        nose.tools.eq_(len(samples), size)
        for s in samples:
            nose.tools.eq_(len(s), 4)
            assert isinstance(s[0], str)
            nose.tools.eq_(s[1].shape, (3, 608, 704)) #planes, height, width
            nose.tools.eq_(s[1].dtype, torch.float32)
            nose.tools.eq_(s[2].shape, (1, 608, 704)) #planes, height, width
            nose.tools.eq_(s[2].dtype, torch.float32)
            nose.tools.eq_(s[3].shape, (1, 608, 704)) #planes, height, width
            nose.tools.eq_(s[3].dtype, torch.float32)
            assert s[1].max() <= 1.0
            assert s[1].min() >= 0.0

    # hack to allow testing on the CI
    from ..configs.datasets.stare import _maker

    for protocol in "ah", "vk":
        dataset = _maker(protocol, stare_dataset)
        nose.tools.eq_(len(dataset), 4)
        _check_subset(dataset["__train__"], 10)
        _check_subset(dataset["train"], 10)
        _check_subset(dataset["test"], 10)


@rc_variable_set("bob.ip.binseg.drive.datadir")
@stare_variable_set("bob.ip.binseg.stare.datadir")
@rc_variable_set("bob.ip.binseg.chasedb1.datadir")
@rc_variable_set("bob.ip.binseg.hrf.datadir")
@rc_variable_set("bob.ip.binseg.iostar.datadir")
def test_stare_mtest():

    from ..configs.datasets.stare.mtest import dataset
    nose.tools.eq_(len(dataset), 10)

    from ..configs.datasets.stare.ah import dataset as baseline
    nose.tools.eq_(dataset["train"], baseline["train"])
    nose.tools.eq_(dataset["test"], baseline["test"])

    for subset in dataset:
        for sample in dataset[subset]:
            assert len(sample) == 4
            assert isinstance(sample[0], str)
            nose.tools.eq_(sample[1].shape, (3, 608, 704)) #planes,height,width
            nose.tools.eq_(sample[1].dtype, torch.float32)
            nose.tools.eq_(sample[2].shape, (1, 608, 704)) #planes,height,width
            nose.tools.eq_(sample[2].dtype, torch.float32)
            nose.tools.eq_(sample[3].shape, (1, 608, 704))
            nose.tools.eq_(sample[3].dtype, torch.float32)
            assert sample[1].max() <= 1.0
            assert sample[1].min() >= 0.0


@stare_variable_set("bob.ip.binseg.stare.datadir")
@rc_variable_set("bob.ip.binseg.drive.datadir")
@rc_variable_set("bob.ip.binseg.chasedb1.datadir")
@rc_variable_set("bob.ip.binseg.hrf.datadir")
@rc_variable_set("bob.ip.binseg.iostar.datadir")
def test_stare_covd():

    from ..configs.datasets.stare.covd import dataset
    nose.tools.eq_(len(dataset), 4)

    from ..configs.datasets.stare.ah import dataset as baseline
    nose.tools.eq_(dataset["train"], dataset["__valid__"])
    nose.tools.eq_(dataset["test"], baseline["test"])

    # these are the only different sets from the baseline
    for key in ("__train__", "train"):
        nose.tools.eq_(len(dataset[key]), 143)
        for sample in dataset[key]:
            assert len(sample) == 4
            assert isinstance(sample[0], str)
            nose.tools.eq_(sample[1].shape, (3, 608, 704)) #planes, height, width
            nose.tools.eq_(sample[1].dtype, torch.float32)
            nose.tools.eq_(sample[2].shape, (1, 608, 704)) #planes, height, width
            nose.tools.eq_(sample[2].dtype, torch.float32)
            assert sample[1].max() <= 1.0
            assert sample[1].min() >= 0.0
            nose.tools.eq_(sample[3].shape, (1, 608, 704))
            nose.tools.eq_(sample[3].dtype, torch.float32)


@rc_variable_set("bob.ip.binseg.chasedb1.datadir")
def test_chasedb1():

    def _check_subset(samples, size):
        nose.tools.eq_(len(samples), size)
        for s in samples:
            nose.tools.eq_(len(s), 4)
            assert isinstance(s[0], str)
            nose.tools.eq_(s[1].shape, (3, 960, 960)) #planes, height, width
            nose.tools.eq_(s[1].dtype, torch.float32)
            nose.tools.eq_(s[2].shape, (1, 960, 960)) #planes, height, width
            nose.tools.eq_(s[2].dtype, torch.float32)
            nose.tools.eq_(s[3].shape, (1, 960, 960)) #planes, height, width
            nose.tools.eq_(s[3].dtype, torch.float32)
            assert s[1].max() <= 1.0
            assert s[1].min() >= 0.0

    for m in ("first_annotator", "second_annotator"):
        d = importlib.import_module(f"...configs.datasets.chasedb1.{m}",
                package=__name__).dataset
        nose.tools.eq_(len(d), 4)
        _check_subset(d["__train__"], 8)
        _check_subset(d["__valid__"], 8)
        _check_subset(d["train"], 8)
        _check_subset(d["test"], 20)


@rc_variable_set("bob.ip.binseg.drive.datadir")
@stare_variable_set("bob.ip.binseg.stare.datadir")
@rc_variable_set("bob.ip.binseg.chasedb1.datadir")
@rc_variable_set("bob.ip.binseg.hrf.datadir")
@rc_variable_set("bob.ip.binseg.iostar.datadir")
def test_chasedb1_mtest():

    from ..configs.datasets.chasedb1.mtest import dataset
    nose.tools.eq_(len(dataset), 10)

    from ..configs.datasets.chasedb1.first_annotator import dataset as baseline
    nose.tools.eq_(dataset["train"], baseline["train"])
    nose.tools.eq_(dataset["test"], baseline["test"])

    for subset in dataset:
        for sample in dataset[subset]:
            assert len(sample) == 4
            assert isinstance(sample[0], str)
            nose.tools.eq_(sample[1].shape, (3, 960, 960)) #planes,height,width
            nose.tools.eq_(sample[1].dtype, torch.float32)
            nose.tools.eq_(sample[2].shape, (1, 960, 960)) #planes,height,width
            nose.tools.eq_(sample[2].dtype, torch.float32)
            nose.tools.eq_(sample[3].shape, (1, 960, 960))
            nose.tools.eq_(sample[3].dtype, torch.float32)
            assert sample[1].max() <= 1.0
            assert sample[1].min() >= 0.0


@rc_variable_set("bob.ip.binseg.drive.datadir")
@stare_variable_set("bob.ip.binseg.stare.datadir")
@rc_variable_set("bob.ip.binseg.chasedb1.datadir")
@rc_variable_set("bob.ip.binseg.hrf.datadir")
@rc_variable_set("bob.ip.binseg.iostar.datadir")
def test_chasedb1_covd():

    from ..configs.datasets.chasedb1.covd import dataset
    nose.tools.eq_(len(dataset), 4)

    from ..configs.datasets.chasedb1.first_annotator import dataset as baseline
    nose.tools.eq_(dataset["train"], dataset["__valid__"])
    nose.tools.eq_(dataset["test"], baseline["test"])

    # these are the only different sets from the baseline
    for key in ("__train__", "train"):
        nose.tools.eq_(len(dataset[key]), 135)
        for sample in dataset[key]:
            assert len(sample) == 4
            assert isinstance(sample[0], str)
            nose.tools.eq_(sample[1].shape, (3, 960, 960)) #planes, height, width
            nose.tools.eq_(sample[1].dtype, torch.float32)
            nose.tools.eq_(sample[2].shape, (1, 960, 960)) #planes, height, width
            nose.tools.eq_(sample[2].dtype, torch.float32)
            nose.tools.eq_(sample[3].shape, (1, 960, 960))
            nose.tools.eq_(sample[3].dtype, torch.float32)
            assert sample[1].max() <= 1.0
            assert sample[1].min() >= 0.0


@rc_variable_set("bob.ip.binseg.hrf.datadir")
def test_hrf():

    def _check_subset(samples, size):
        nose.tools.eq_(len(samples), size)
        for s in samples:
            nose.tools.eq_(len(s), 4)
            assert isinstance(s[0], str)
            nose.tools.eq_(s[1].shape, (3, 1168, 1648)) #planes, height, width
            nose.tools.eq_(s[1].dtype, torch.float32)
            nose.tools.eq_(s[2].shape, (1, 1168, 1648)) #planes, height, width
            nose.tools.eq_(s[2].dtype, torch.float32)
            nose.tools.eq_(s[3].shape, (1, 1168, 1648)) #planes, height, width
            nose.tools.eq_(s[3].dtype, torch.float32)
            assert s[1].max() <= 1.0
            assert s[1].min() >= 0.0

    def _check_subset_fullres(samples, size):
        nose.tools.eq_(len(samples), size)
        for s in samples:
            nose.tools.eq_(len(s), 4)
            assert isinstance(s[0], str)
            nose.tools.eq_(s[1].shape, (3, 2336, 3296)) #planes, height, width
            nose.tools.eq_(s[1].dtype, torch.float32)
            nose.tools.eq_(s[2].shape, (1, 2336, 3296)) #planes, height, width
            nose.tools.eq_(s[2].dtype, torch.float32)
            nose.tools.eq_(s[3].shape, (1, 2336, 3296)) #planes, height, width
            nose.tools.eq_(s[3].dtype, torch.float32)
            assert s[1].max() <= 1.0
            assert s[1].min() >= 0.0

    from ..configs.datasets.hrf.default import dataset
    nose.tools.eq_(len(dataset), 6)
    _check_subset(dataset["__train__"], 15)
    _check_subset(dataset["train"], 15)
    _check_subset(dataset["test"], 30)
    _check_subset_fullres(dataset["train (full resolution)"], 15)
    _check_subset_fullres(dataset["test (full resolution)"], 30)


@rc_variable_set("bob.ip.binseg.drive.datadir")
@stare_variable_set("bob.ip.binseg.stare.datadir")
@rc_variable_set("bob.ip.binseg.chasedb1.datadir")
@rc_variable_set("bob.ip.binseg.hrf.datadir")
@rc_variable_set("bob.ip.binseg.iostar.datadir")
def test_hrf_mtest():

    from ..configs.datasets.hrf.mtest import dataset
    nose.tools.eq_(len(dataset), 12)

    from ..configs.datasets.hrf.default import dataset as baseline
    nose.tools.eq_(dataset["train"], baseline["train"])
    nose.tools.eq_(dataset["test"], baseline["test"])

    for subset in dataset:
        for sample in dataset[subset]:
            assert len(sample) == 4
            assert isinstance(sample[0], str)
            if "full resolution" in subset:
                nose.tools.eq_(sample[1].shape, (3, 2336, 3296))
                nose.tools.eq_(sample[1].dtype, torch.float32)
                nose.tools.eq_(sample[2].shape, (1, 2336, 3296))
                nose.tools.eq_(sample[2].dtype, torch.float32)
                nose.tools.eq_(sample[3].shape, (1, 2336, 3296))
                nose.tools.eq_(sample[3].dtype, torch.float32)
            else:
                nose.tools.eq_(sample[1].shape, (3, 1168, 1648))
                nose.tools.eq_(sample[1].dtype, torch.float32)
                nose.tools.eq_(sample[2].shape, (1, 1168, 1648))
                nose.tools.eq_(sample[2].dtype, torch.float32)
                nose.tools.eq_(sample[3].shape, (1, 1168, 1648))
                nose.tools.eq_(sample[3].dtype, torch.float32)
            assert sample[1].max() <= 1.0
            assert sample[1].min() >= 0.0


@rc_variable_set("bob.ip.binseg.drive.datadir")
@stare_variable_set("bob.ip.binseg.stare.datadir")
@rc_variable_set("bob.ip.binseg.chasedb1.datadir")
@rc_variable_set("bob.ip.binseg.hrf.datadir")
@rc_variable_set("bob.ip.binseg.iostar.datadir")
def test_hrf_covd():

    from ..configs.datasets.hrf.covd import dataset
    nose.tools.eq_(len(dataset), 6)

    from ..configs.datasets.hrf.default import dataset as baseline
    nose.tools.eq_(dataset["train"], dataset["__valid__"])
    nose.tools.eq_(dataset["test"], baseline["test"])

    # these are the only different sets from the baseline
    for key in ("__train__", "train"):
        nose.tools.eq_(len(dataset[key]), 118)
        for sample in dataset[key]:
            assert len(sample) == 4
            assert isinstance(sample[0], str)
            nose.tools.eq_(sample[1].shape, (3, 1168, 1648))
            nose.tools.eq_(sample[1].dtype, torch.float32)
            nose.tools.eq_(sample[2].shape, (1, 1168, 1648))
            nose.tools.eq_(sample[2].dtype, torch.float32)
            nose.tools.eq_(sample[3].shape, (1, 1168, 1648))
            nose.tools.eq_(sample[3].dtype, torch.float32)
            assert sample[1].max() <= 1.0
            assert sample[1].min() >= 0.0


@rc_variable_set("bob.ip.binseg.iostar.datadir")
def test_iostar():

    def _check_subset(samples, size):
        nose.tools.eq_(len(samples), size)
        for s in samples:
            nose.tools.eq_(len(s), 4)
            assert isinstance(s[0], str)
            nose.tools.eq_(s[1].shape, (3, 1024, 1024)) #planes, height, width
            nose.tools.eq_(s[1].dtype, torch.float32)
            nose.tools.eq_(s[2].shape, (1, 1024, 1024)) #planes, height, width
            nose.tools.eq_(s[2].dtype, torch.float32)
            nose.tools.eq_(s[3].shape, (1, 1024, 1024)) #planes, height, width
            nose.tools.eq_(s[3].dtype, torch.float32)
            assert s[1].max() <= 1.0
            assert s[1].min() >= 0.0

    for m in ("vessel", "optic_disc"):
        d = importlib.import_module(f"...configs.datasets.iostar.{m}",
                package=__name__).dataset
        nose.tools.eq_(len(d), 4)
        _check_subset(d["__train__"], 20)
        _check_subset(d["train"], 20)
        _check_subset(d["test"], 10)


@rc_variable_set("bob.ip.binseg.drive.datadir")
@stare_variable_set("bob.ip.binseg.stare.datadir")
@rc_variable_set("bob.ip.binseg.chasedb1.datadir")
@rc_variable_set("bob.ip.binseg.hrf.datadir")
@rc_variable_set("bob.ip.binseg.iostar.datadir")
def test_iostar_mtest():

    from ..configs.datasets.iostar.vessel_mtest import dataset
    nose.tools.eq_(len(dataset), 10)

    from ..configs.datasets.iostar.vessel import dataset as baseline
    nose.tools.eq_(dataset["train"], baseline["train"])
    nose.tools.eq_(dataset["test"], baseline["test"])

    for subset in dataset:
        for sample in dataset[subset]:
            assert len(sample) == 4
            assert isinstance(sample[0], str)
            nose.tools.eq_(sample[1].shape, (3, 1024, 1024)) #planes,height,width
            nose.tools.eq_(sample[1].dtype, torch.float32)
            nose.tools.eq_(sample[2].shape, (1, 1024, 1024)) #planes,height,width
            nose.tools.eq_(sample[2].dtype, torch.float32)
            nose.tools.eq_(sample[3].shape, (1, 1024, 1024))
            nose.tools.eq_(sample[3].dtype, torch.float32)
            assert sample[1].max() <= 1.0
            assert sample[1].min() >= 0.0


@rc_variable_set("bob.ip.binseg.drive.datadir")
@stare_variable_set("bob.ip.binseg.stare.datadir")
@rc_variable_set("bob.ip.binseg.chasedb1.datadir")
@rc_variable_set("bob.ip.binseg.hrf.datadir")
@rc_variable_set("bob.ip.binseg.iostar.datadir")
def test_iostar_covd():

    from ..configs.datasets.iostar.covd import dataset
    nose.tools.eq_(len(dataset), 4)

    from ..configs.datasets.iostar.vessel import dataset as baseline
    nose.tools.eq_(dataset["train"], dataset["__valid__"])
    nose.tools.eq_(dataset["test"], baseline["test"])

    # these are the only different sets from the baseline
    for key in ("__train__", "train"):
        nose.tools.eq_(len(dataset[key]), 133)
        for sample in dataset[key]:
            assert len(sample) == 4
            assert isinstance(sample[0], str)
            nose.tools.eq_(sample[1].shape, (3, 1024, 1024))
            nose.tools.eq_(sample[1].dtype, torch.float32)
            nose.tools.eq_(sample[2].shape, (1, 1024, 1024))
            nose.tools.eq_(sample[2].dtype, torch.float32)
            nose.tools.eq_(sample[3].shape, (1, 1024, 1024))
            nose.tools.eq_(sample[3].dtype, torch.float32)
            assert sample[1].max() <= 1.0
            assert sample[1].min() >= 0.0


@rc_variable_set("bob.ip.binseg.refuge.datadir")
def test_refuge():

    def _check_subset(samples, size):
        nose.tools.eq_(len(samples), size)
        for s in samples[:N]:
            nose.tools.eq_(len(s), 3)
            assert isinstance(s[0], str)
            nose.tools.eq_(s[1].shape, (3, 1632, 1632)) #planes, height, width
            nose.tools.eq_(s[1].dtype, torch.float32)
            nose.tools.eq_(s[2].shape, (1, 1632, 1632)) #planes, height, width
            nose.tools.eq_(s[2].dtype, torch.float32)
            assert s[1].max() <= 1.0
            assert s[1].min() >= 0.0

    for m in ("disc", "cup"):
        d = importlib.import_module(f"...configs.datasets.refuge.{m}",
                package=__name__).dataset
        nose.tools.eq_(len(d), 5)
        _check_subset(d["__train__"], 400)
        _check_subset(d["train"], 400)
        _check_subset(d["validation"], 400)
        _check_subset(d["test"], 400)


@rc_variable_set("bob.ip.binseg.drishtigs1.datadir")
def test_drishtigs1():

    def _check_subset(samples, size):
        nose.tools.eq_(len(samples), size)
        for s in samples[:N]:
            nose.tools.eq_(len(s), 3)
            assert isinstance(s[0], str)
            nose.tools.eq_(s[1].shape, (3, 1760, 2048)) #planes, height, width
            nose.tools.eq_(s[1].dtype, torch.float32)
            nose.tools.eq_(s[2].shape, (1, 1760, 2048)) #planes, height, width
            nose.tools.eq_(s[2].dtype, torch.float32)
            assert s[1].max() <= 1.0
            assert s[1].min() >= 0.0

    for m in ("disc_all", "cup_all", "disc_any", "cup_any"):
        d = importlib.import_module(f"...configs.datasets.drishtigs1.{m}",
                package=__name__).dataset
        nose.tools.eq_(len(d), 4)
        _check_subset(d["__train__"], 50)
        _check_subset(d["train"], 50)
        _check_subset(d["test"], 51)


@rc_variable_set("bob.ip.binseg.rimoner3.datadir")
def test_rimoner3():

    def _check_subset(samples, size):
        nose.tools.eq_(len(samples), size)
        for s in samples[:N]:
            nose.tools.eq_(len(s), 3)
            assert isinstance(s[0], str)
            nose.tools.eq_(s[1].shape, (3, 1440, 1088)) #planes, height, width
            nose.tools.eq_(s[1].dtype, torch.float32)
            nose.tools.eq_(s[2].shape, (1, 1440, 1088)) #planes, height, width
            nose.tools.eq_(s[2].dtype, torch.float32)
            assert s[1].max() <= 1.0
            assert s[1].min() >= 0.0

    for m in ("disc_exp1", "cup_exp1", "disc_exp2", "cup_exp2"):
        d = importlib.import_module(f"...configs.datasets.rimoner3.{m}",
                package=__name__).dataset
        nose.tools.eq_(len(d), 4)
        _check_subset(d["__train__"], 99)
        _check_subset(d["train"], 99)
        _check_subset(d["test"], 60)


@rc_variable_set("bob.ip.binseg.drionsdb.datadir")
def test_drionsdb():

    def _check_subset(samples, size):
        nose.tools.eq_(len(samples), size)
        for s in samples[:N]:
            nose.tools.eq_(len(s), 3)
            assert isinstance(s[0], str)
            nose.tools.eq_(s[1].shape, (3, 416, 608)) #planes, height, width
            nose.tools.eq_(s[1].dtype, torch.float32)
            nose.tools.eq_(s[2].shape, (1, 416, 608)) #planes, height, width
            nose.tools.eq_(s[2].dtype, torch.float32)
            assert s[1].max() <= 1.0
            assert s[1].min() >= 0.0

    for m in ("expert1", "expert2"):
        d = importlib.import_module(f"...configs.datasets.drionsdb.{m}",
                package=__name__).dataset
        nose.tools.eq_(len(d), 4)
        _check_subset(d["__train__"], 60)
        _check_subset(d["train"], 60)
        _check_subset(d["test"], 50)
