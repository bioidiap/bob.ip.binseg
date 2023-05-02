# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

import importlib

import pytest
import torch

# we only iterate over the first N elements at most - dataset loading has
# already been checked on the individual datset tests.  Here, we are only
# testing for the extra tools wrapping the dataset
N = 10


@pytest.mark.skip_if_rc_var_not_set("datadir.drive")
def test_drive():
    def _check_subset(samples, size, height, width):
        assert len(samples) == size
        for s in samples:
            assert len(s) == 4
            assert isinstance(s[0], str)
            assert s[1].shape, (3, height == height)  # planes, height, width
            assert s[1].dtype == torch.float32
            assert s[2].shape, (1, height == width)  # planes, height, width
            assert s[2].dtype == torch.float32
            assert s[3].shape, (1, height == width)  # planes, height, width
            assert s[3].dtype == torch.float32
            assert s[1].max() <= 1.0
            assert s[1].min() >= 0.0

    from deepdraw.configs.datasets.drive.default import dataset

    assert len(dataset) == 4
    _check_subset(dataset["__train__"], 20, 544, 544)
    _check_subset(dataset["__valid__"], 20, 544, 544)
    _check_subset(dataset["train"], 20, 544, 544)
    _check_subset(dataset["test"], 20, 544, 544)

    from deepdraw.configs.datasets.drive.second_annotator import dataset

    assert len(dataset) == 1
    _check_subset(dataset["test"], 20, 544, 544)

    from deepdraw.configs.datasets.drive.default_768 import dataset

    _check_subset(dataset["__train__"], 20, 768, 768)
    _check_subset(dataset["__valid__"], 20, 768, 768)
    _check_subset(dataset["train"], 20, 768, 768)
    _check_subset(dataset["test"], 20, 768, 768)

    from deepdraw.configs.datasets.drive.default_1024 import dataset

    _check_subset(dataset["__train__"], 20, 1024, 1024)
    _check_subset(dataset["__valid__"], 20, 1024, 1024)
    _check_subset(dataset["train"], 20, 1024, 1024)
    _check_subset(dataset["test"], 20, 1024, 1024)


@pytest.mark.skip_if_rc_var_not_set("datadir.drive")
@pytest.mark.skip_if_rc_var_not_set("datadir.chasedb1")
@pytest.mark.skip_if_rc_var_not_set("datadir.hrf")
@pytest.mark.skip_if_rc_var_not_set("datadir.iostar")
def test_drive_mtest():
    from deepdraw.configs.datasets.drive.mtest import dataset

    assert len(dataset) == 10

    from deepdraw.configs.datasets.drive.default import dataset as baseline

    assert dataset["train"] == baseline["train"]
    assert dataset["test"] == baseline["test"]

    for subset in dataset:
        for sample in dataset[subset]:
            assert len(sample) == 4
            assert isinstance(sample[0], str)
            assert sample[1].shape, (3, 544 == 544)  # planes, height, width
            assert sample[1].dtype == torch.float32
            assert sample[2].shape, (1, 544 == 544)
            assert sample[2].dtype == torch.float32
            assert sample[3].shape, (1, 544 == 544)
            assert sample[3].dtype == torch.float32
            assert sample[1].max() <= 1.0
            assert sample[1].min() >= 0.0


@pytest.mark.skip_if_rc_var_not_set("datadir.drive")
@pytest.mark.skip_if_rc_var_not_set("datadir.chasedb1")
@pytest.mark.skip_if_rc_var_not_set("datadir.hrf")
@pytest.mark.skip_if_rc_var_not_set("datadir.iostar")
def test_drive_covd():
    from deepdraw.configs.datasets.drive.covd import dataset

    assert len(dataset) == 4

    from deepdraw.configs.datasets.drive.default import dataset as baseline

    assert dataset["train"] == dataset["__valid__"]
    assert dataset["test"] == baseline["test"]

    for key in ("__train__", "train"):
        assert len(dataset[key]) == 123
        for sample in dataset["__train__"]:
            assert len(sample) == 4
            assert isinstance(sample[0], str)
            assert sample[1].shape, (3, 544 == 544)  # planes, height, width
            assert sample[1].dtype == torch.float32
            assert sample[2].shape, (1, 544 == 544)  # planes, height, width
            assert sample[2].dtype == torch.float32
            assert sample[3].shape, (1, 544 == 544)
            assert sample[3].dtype == torch.float32
            assert sample[1].max() <= 1.0
            assert sample[1].min() >= 0.0


@pytest.mark.skip_if_rc_var_not_set("datadir.stare")
def test_stare_augmentation_manipulation():
    # some tests to check our context management for dataset augmentation works
    # adequately, with one example dataset

    # hack to allow testing on the CI
    from deepdraw.configs.datasets.stare.ah import dataset

    assert len(dataset["__train__"]._transforms.transforms) == (
        len(dataset["test"]._transforms.transforms) + 4
    )

    assert len(dataset["train"]._transforms.transforms) == len(
        dataset["test"]._transforms.transforms
    )


@pytest.mark.skip_if_rc_var_not_set("datadir.stare")
def test_stare():
    def _check_subset(samples, size, height, width):
        assert len(samples) == size
        for s in samples:
            assert len(s) == 4
            assert isinstance(s[0], str)
            assert s[1].shape, (3, height == width)  # planes, height, width
            assert s[1].dtype == torch.float32
            assert s[2].shape, (1, height == width)  # planes, height, width
            assert s[2].dtype == torch.float32
            assert s[3].shape, (1, height == width)  # planes, height, width
            assert s[3].dtype == torch.float32
            assert s[1].max() <= 1.0
            assert s[1].min() >= 0.0

    # hack to allow testing on the CI
    from deepdraw.configs.datasets.stare import _maker, _maker_square

    for protocol in "ah", "vk":
        dataset = _maker(protocol)
        assert len(dataset) == 4
        _check_subset(dataset["__train__"], 10, 608, 704)
        _check_subset(dataset["train"], 10, 608, 704)
        _check_subset(dataset["test"], 10, 608, 704)

    dataset = _maker_square("ah", 768)
    assert len(dataset) == 4
    _check_subset(dataset["__train__"], 10, 768, 768)
    _check_subset(dataset["train"], 10, 768, 768)
    _check_subset(dataset["test"], 10, 768, 768)

    dataset = _maker_square("ah", 1024)
    assert len(dataset) == 4
    _check_subset(dataset["__train__"], 10, 1024, 1024)
    _check_subset(dataset["train"], 10, 1024, 1024)
    _check_subset(dataset["test"], 10, 1024, 1024)


@pytest.mark.skip_if_rc_var_not_set("datadir.drive")
@pytest.mark.skip_if_rc_var_not_set("datadir.chasedb1")
@pytest.mark.skip_if_rc_var_not_set("datadir.hrf")
@pytest.mark.skip_if_rc_var_not_set("datadir.iostar")
def test_stare_mtest():
    from deepdraw.configs.datasets.stare.mtest import dataset

    assert len(dataset) == 10

    from deepdraw.configs.datasets.stare.ah import dataset as baseline

    assert dataset["train"] == baseline["train"]
    assert dataset["test"] == baseline["test"]

    for subset in dataset:
        for sample in dataset[subset]:
            assert len(sample) == 4
            assert isinstance(sample[0], str)
            assert sample[1].shape, (3, 608 == 704)  # planes,height,width
            assert sample[1].dtype == torch.float32
            assert sample[2].shape, (1, 608 == 704)  # planes,height,width
            assert sample[2].dtype == torch.float32
            assert sample[3].shape, (1, 608 == 704)
            assert sample[3].dtype == torch.float32
            assert sample[1].max() <= 1.0
            assert sample[1].min() >= 0.0


@pytest.mark.skip_if_rc_var_not_set("datadir.drive")
@pytest.mark.skip_if_rc_var_not_set("datadir.chasedb1")
@pytest.mark.skip_if_rc_var_not_set("datadir.hrf")
@pytest.mark.skip_if_rc_var_not_set("datadir.iostar")
def test_stare_covd():
    from deepdraw.configs.datasets.stare.covd import dataset

    assert len(dataset) == 4

    from deepdraw.configs.datasets.stare.ah import dataset as baseline

    assert dataset["train"] == dataset["__valid__"]
    assert dataset["test"] == baseline["test"]

    # these are the only different sets from the baseline
    for key in ("__train__", "train"):
        assert len(dataset[key]) == 143
        for sample in dataset[key]:
            assert len(sample) == 4
            assert isinstance(sample[0], str)
            assert sample[1].shape, (3, 608 == 704)  # planes, height, width
            assert sample[1].dtype == torch.float32
            assert sample[2].shape, (1, 608 == 704)  # planes, height, width
            assert sample[2].dtype == torch.float32
            assert sample[1].max() <= 1.0
            assert sample[1].min() >= 0.0
            assert sample[3].shape, (1, 608 == 704)
            assert sample[3].dtype == torch.float32


@pytest.mark.skip_if_rc_var_not_set("datadir.chasedb1")
def test_chasedb1():
    def _check_subset(samples, size, height, width):
        assert len(samples) == size
        for s in samples:
            assert len(s) == 4
            assert isinstance(s[0], str)
            assert s[1].shape, (3, height == width)  # planes, height, width
            assert s[1].dtype == torch.float32
            assert s[2].shape, (1, height == width)  # planes, height, width
            assert s[2].dtype == torch.float32
            assert s[3].shape, (1, height == width)  # planes, height, width
            assert s[3].dtype == torch.float32
            assert s[1].max() <= 1.0
            assert s[1].min() >= 0.0

    for m in ("first_annotator", "second_annotator"):
        d = importlib.import_module(
            f"deepdraw.configs.datasets.chasedb1.{m}", package=__name__
        ).dataset
        assert len(d) == 4
        _check_subset(d["__train__"], 8, 960, 960)
        _check_subset(d["__valid__"], 8, 960, 960)
        _check_subset(d["train"], 8, 960, 960)
        _check_subset(d["test"], 20, 960, 960)

    from deepdraw.configs.datasets.chasedb1.first_annotator_768 import dataset

    assert len(dataset) == 4
    _check_subset(dataset["__train__"], 8, 768, 768)
    _check_subset(dataset["__valid__"], 8, 768, 768)
    _check_subset(dataset["train"], 8, 768, 768)
    _check_subset(dataset["test"], 20, 768, 768)

    from deepdraw.configs.datasets.chasedb1.first_annotator_1024 import dataset

    assert len(dataset) == 4
    _check_subset(dataset["__train__"], 8, 1024, 1024)
    _check_subset(dataset["__valid__"], 8, 1024, 1024)
    _check_subset(dataset["train"], 8, 1024, 1024)
    _check_subset(dataset["test"], 20, 1024, 1024)


@pytest.mark.skip_if_rc_var_not_set("datadir.drive")
@pytest.mark.skip_if_rc_var_not_set("datadir.chasedb1")
@pytest.mark.skip_if_rc_var_not_set("datadir.hrf")
@pytest.mark.skip_if_rc_var_not_set("datadir.iostar")
def test_chasedb1_mtest():
    from deepdraw.configs.datasets.chasedb1.mtest import dataset

    assert len(dataset) == 10

    from deepdraw.configs.datasets.chasedb1.first_annotator import (
        dataset as baseline,
    )

    assert dataset["train"] == baseline["train"]
    assert dataset["test"] == baseline["test"]

    for subset in dataset:
        for sample in dataset[subset]:
            assert len(sample) == 4
            assert isinstance(sample[0], str)
            assert sample[1].shape, (3, 960 == 960)  # planes,height,width
            assert sample[1].dtype == torch.float32
            assert sample[2].shape, (1, 960 == 960)  # planes,height,width
            assert sample[2].dtype == torch.float32
            assert sample[3].shape, (1, 960 == 960)
            assert sample[3].dtype == torch.float32
            assert sample[1].max() <= 1.0
            assert sample[1].min() >= 0.0


@pytest.mark.skip_if_rc_var_not_set("datadir.drive")
@pytest.mark.skip_if_rc_var_not_set("datadir.chasedb1")
@pytest.mark.skip_if_rc_var_not_set("datadir.hrf")
@pytest.mark.skip_if_rc_var_not_set("datadir.iostar")
def test_chasedb1_covd():
    from deepdraw.configs.datasets.chasedb1.covd import dataset

    assert len(dataset) == 4

    from deepdraw.configs.datasets.chasedb1.first_annotator import (
        dataset as baseline,
    )

    assert dataset["train"] == dataset["__valid__"]
    assert dataset["test"] == baseline["test"]

    # these are the only different sets from the baseline
    for key in ("__train__", "train"):
        assert len(dataset[key]) == 135
        for sample in dataset[key]:
            assert len(sample) == 4
            assert isinstance(sample[0], str)
            assert sample[1].shape, (3, 960 == 960)  # planes, height, width
            assert sample[1].dtype == torch.float32
            assert sample[2].shape, (1, 960 == 960)  # planes, height, width
            assert sample[2].dtype == torch.float32
            assert sample[3].shape, (1, 960 == 960)
            assert sample[3].dtype == torch.float32
            assert sample[1].max() <= 1.0
            assert sample[1].min() >= 0.0


@pytest.mark.skip_if_rc_var_not_set("datadir.hrf")
def test_hrf():
    def _check_subset(samples, size, height, width):
        assert len(samples) == size
        for s in samples:
            assert len(s) == 4
            assert isinstance(s[0], str)
            assert s[1].shape, (3, height == width)  # planes, height, width
            assert s[1].dtype == torch.float32
            assert s[2].shape, (1, height == width)  # planes, height, width
            assert s[2].dtype == torch.float32
            assert s[3].shape, (1, height == width)  # planes, height, width
            assert s[3].dtype == torch.float32
            assert s[1].max() <= 1.0
            assert s[1].min() >= 0.0

    def _check_subset_fullres(samples, size):
        assert len(samples) == size
        for s in samples:
            assert len(s) == 4
            assert isinstance(s[0], str)
            assert s[1].shape, (3, 2336 == 3296)  # planes, height, width
            assert s[1].dtype == torch.float32
            assert s[2].shape, (1, 2336 == 3296)  # planes, height, width
            assert s[2].dtype == torch.float32
            assert s[3].shape, (1, 2336 == 3296)  # planes, height, width
            assert s[3].dtype == torch.float32
            assert s[1].max() <= 1.0
            assert s[1].min() >= 0.0

    from deepdraw.configs.datasets.hrf.default import dataset

    assert len(dataset) == 6
    _check_subset(dataset["__train__"], 15, 1168, 1648)
    _check_subset(dataset["train"], 15, 1168, 1648)
    _check_subset(dataset["test"], 30, 1168, 1648)
    _check_subset_fullres(dataset["train (full resolution)"], 15)
    _check_subset_fullres(dataset["test (full resolution)"], 30)

    from deepdraw.configs.datasets.hrf.default_768 import dataset

    assert len(dataset) == 4
    _check_subset(dataset["__train__"], 15, 768, 768)
    _check_subset(dataset["train"], 15, 768, 768)
    _check_subset(dataset["test"], 30, 768, 768)

    from deepdraw.configs.datasets.hrf.default_1024 import dataset

    assert len(dataset) == 4
    _check_subset(dataset["__train__"], 15, 1024, 1024)
    _check_subset(dataset["train"], 15, 1024, 1024)
    _check_subset(dataset["test"], 30, 1024, 1024)


@pytest.mark.skip_if_rc_var_not_set("datadir.drive")
@pytest.mark.skip_if_rc_var_not_set("datadir.chasedb1")
@pytest.mark.skip_if_rc_var_not_set("datadir.hrf")
@pytest.mark.skip_if_rc_var_not_set("datadir.iostar")
def test_hrf_mtest():
    from deepdraw.configs.datasets.hrf.mtest import dataset

    assert len(dataset) == 12

    from deepdraw.configs.datasets.hrf.default import dataset as baseline

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


@pytest.mark.skip_if_rc_var_not_set("datadir.drive")
@pytest.mark.skip_if_rc_var_not_set("datadir.chasedb1")
@pytest.mark.skip_if_rc_var_not_set("datadir.hrf")
@pytest.mark.skip_if_rc_var_not_set("datadir.iostar")
def test_hrf_covd():
    from deepdraw.configs.datasets.hrf.covd import dataset

    assert len(dataset) == 6

    from deepdraw.configs.datasets.hrf.default import dataset as baseline

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


@pytest.mark.skip_if_rc_var_not_set("datadir.iostar")
def test_iostar():
    def _check_subset(samples, size, height, width):
        assert len(samples) == size
        for s in samples:
            assert len(s) == 4
            assert isinstance(s[0], str)
            assert s[1].shape, (3, height == width)  # planes, height, width
            assert s[1].dtype == torch.float32
            assert s[2].shape, (1, height == width)  # planes, height, width
            assert s[2].dtype == torch.float32
            assert s[3].shape, (1, height == width)  # planes, height, width
            assert s[3].dtype == torch.float32
            assert s[1].max() <= 1.0
            assert s[1].min() >= 0.0

    for m in ("vessel", "optic_disc"):
        d = importlib.import_module(
            f"deepdraw.configs.datasets.iostar.{m}", package=__name__
        ).dataset
        assert len(d) == 4
        _check_subset(d["__train__"], 20, 1024, 1024)
        _check_subset(d["train"], 20, 1024, 1024)
        _check_subset(d["test"], 10, 1024, 1024)

    for m in ("vessel_768", "optic_disc_768"):
        d = importlib.import_module(
            f"deepdraw.configs.datasets.iostar.{m}", package=__name__
        ).dataset
        assert len(d) == 4
        _check_subset(d["__train__"], 20, 768, 768)
        _check_subset(d["train"], 20, 768, 768)
        _check_subset(d["test"], 10, 768, 768)

    from deepdraw.configs.datasets.iostar.optic_disc_512 import dataset

    assert len(dataset) == 4
    _check_subset(dataset["__train__"], 20, 512, 512)
    _check_subset(dataset["train"], 20, 512, 512)
    _check_subset(dataset["test"], 10, 512, 512)


@pytest.mark.skip_if_rc_var_not_set("datadir.drive")
@pytest.mark.skip_if_rc_var_not_set("datadir.chasedb1")
@pytest.mark.skip_if_rc_var_not_set("datadir.hrf")
@pytest.mark.skip_if_rc_var_not_set("datadir.iostar")
def test_iostar_mtest():
    from deepdraw.configs.datasets.iostar.vessel_mtest import dataset

    assert len(dataset) == 10

    from deepdraw.configs.datasets.iostar.vessel import dataset as baseline

    assert dataset["train"] == baseline["train"]
    assert dataset["test"] == baseline["test"]

    for subset in dataset:
        for sample in dataset[subset]:
            assert len(sample) == 4
            assert isinstance(sample[0], str)
            assert sample[1].shape, (3, 1024 == 1024)  # planes,height,width
            assert sample[1].dtype == torch.float32
            assert sample[2].shape, (1, 1024 == 1024)  # planes,height,width
            assert sample[2].dtype == torch.float32
            assert sample[3].shape, (1, 1024 == 1024)
            assert sample[3].dtype == torch.float32
            assert sample[1].max() <= 1.0
            assert sample[1].min() >= 0.0


@pytest.mark.skip_if_rc_var_not_set("datadir.drive")
@pytest.mark.skip_if_rc_var_not_set("datadir.chasedb1")
@pytest.mark.skip_if_rc_var_not_set("datadir.hrf")
@pytest.mark.skip_if_rc_var_not_set("datadir.iostar")
def test_iostar_covd():
    from deepdraw.configs.datasets.iostar.covd import dataset

    assert len(dataset) == 4

    from deepdraw.configs.datasets.iostar.vessel import dataset as baseline

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


@pytest.mark.skip_if_rc_var_not_set("datadir.refuge")
def test_refuge():
    def _check_subset(samples, size, height, width):
        assert len(samples) == size
        for s in samples[:N]:
            assert len(s) == 4
            assert isinstance(s[0], str)
            assert s[1].shape, (3, height == width)  # planes, height, width
            assert s[1].dtype == torch.float32
            assert s[2].shape, (1, height == width)  # planes, height, width
            assert s[2].dtype == torch.float32
            assert s[1].max() <= 1.0
            assert s[1].min() >= 0.0

    for m in ("disc", "cup"):
        d = importlib.import_module(
            f"deepdraw.configs.datasets.refuge.{m}", package=__name__
        ).dataset
        assert len(d) == 5
        _check_subset(d["__train__"], 400, 1632, 1632)
        _check_subset(d["train"], 400, 1632, 1632)
        _check_subset(d["validation"], 400, 1632, 1632)
        _check_subset(d["test"], 400, 1632, 1632)

    for m in ("disc_512", "cup_512"):
        d = importlib.import_module(
            f"deepdraw.configs.datasets.refuge.{m}", package=__name__
        ).dataset
        assert len(d) == 5
        _check_subset(d["__train__"], 400, 512, 512)
        _check_subset(d["train"], 400, 512, 512)
        _check_subset(d["validation"], 400, 512, 512)
        _check_subset(d["test"], 400, 512, 512)

    for m in ("disc_768", "cup_768"):
        d = importlib.import_module(
            f"deepdraw.configs.datasets.refuge.{m}", package=__name__
        ).dataset
        assert len(d) == 5
        _check_subset(d["__train__"], 400, 768, 768)
        _check_subset(d["train"], 400, 768, 768)
        _check_subset(d["validation"], 400, 768, 768)
        _check_subset(d["test"], 400, 768, 768)


@pytest.mark.skip_if_rc_var_not_set("datadir.drishtigs1")
def test_drishtigs1():
    def _check_subset(samples, size, height, width):
        assert len(samples) == size
        for s in samples[:N]:
            assert len(s) == 4
            assert isinstance(s[0], str)
            assert s[1].shape, (3, height == width)  # planes, height, width
            assert s[1].dtype == torch.float32
            assert s[2].shape, (1, height == width)  # planes, height, width
            assert s[2].dtype == torch.float32
            assert s[1].max() <= 1.0
            assert s[1].min() >= 0.0

    for m in ("disc_all", "cup_all", "disc_any", "cup_any"):
        d = importlib.import_module(
            f"deepdraw.configs.datasets.drishtigs1.{m}", package=__name__
        ).dataset
        assert len(d) == 4
        _check_subset(d["__train__"], 50, 1760, 2048)
        _check_subset(d["train"], 50, 1760, 2048)
        _check_subset(d["test"], 51, 1760, 2048)

    for m in ("disc_all_512", "cup_all_512"):
        d = importlib.import_module(
            f"deepdraw.configs.datasets.drishtigs1.{m}", package=__name__
        ).dataset
        assert len(d) == 4
        _check_subset(d["__train__"], 50, 512, 512)
        _check_subset(d["train"], 50, 512, 512)
        _check_subset(d["test"], 51, 512, 512)
    for m in ("disc_all_768", "cup_all_768"):
        d = importlib.import_module(
            f"deepdraw.configs.datasets.drishtigs1.{m}", package=__name__
        ).dataset
        assert len(d) == 4
        _check_subset(d["__train__"], 50, 768, 768)
        _check_subset(d["train"], 50, 768, 768)
        _check_subset(d["test"], 51, 768, 768)


@pytest.mark.skip_if_rc_var_not_set("datadir.rimoner3")
def test_rimoner3():
    def _check_subset(samples, size, height, width):
        assert len(samples) == size
        for s in samples[:N]:
            assert len(s) == 4
            assert isinstance(s[0], str)
            assert s[1].shape, (3, height == width)  # planes, height, width
            assert s[1].dtype == torch.float32
            assert s[2].shape, (1, height == width)  # planes, height, width
            assert s[2].dtype == torch.float32
            assert s[1].max() <= 1.0
            assert s[1].min() >= 0.0

    for m in ("disc_exp1", "cup_exp1", "disc_exp2", "cup_exp2"):
        d = importlib.import_module(
            f"deepdraw.configs.datasets.rimoner3.{m}", package=__name__
        ).dataset
        assert len(d) == 4
        _check_subset(d["__train__"], 99, 1440, 1088)
        _check_subset(d["train"], 99, 1440, 1088)
        _check_subset(d["test"], 60, 1440, 1088)

    for m in ("disc_exp1_512", "cup_exp1_512"):
        d = importlib.import_module(
            f"deepdraw.configs.datasets.rimoner3.{m}", package=__name__
        ).dataset
        assert len(d) == 4
        _check_subset(d["__train__"], 99, 512, 512)
        _check_subset(d["train"], 99, 512, 512)
        _check_subset(d["test"], 60, 512, 512)

    for m in ("disc_exp1_768", "cup_exp1_768"):
        d = importlib.import_module(
            f"deepdraw.configs.datasets.rimoner3.{m}", package=__name__
        ).dataset
        assert len(d) == 4
        _check_subset(d["__train__"], 99, 768, 768)
        _check_subset(d["train"], 99, 768, 768)
        _check_subset(d["test"], 60, 768, 768)


@pytest.mark.skip_if_rc_var_not_set("datadir.drionsdb")
def test_drionsdb():
    def _check_subset(samples, size, height, width):
        assert len(samples) == size
        for s in samples[:N]:
            assert len(s) == 4
            assert isinstance(s[0], str)
            assert s[1].shape, (3, height == width)  # planes, height, width
            assert s[1].dtype == torch.float32
            assert s[2].shape, (1, height == width)  # planes, height, width
            assert s[2].dtype == torch.float32
            assert s[1].max() <= 1.0
            assert s[1].min() >= 0.0

    for m in ("expert1", "expert2"):
        d = importlib.import_module(
            f"deepdraw.configs.datasets.drionsdb.{m}", package=__name__
        ).dataset
        assert len(d) == 4
        _check_subset(d["__train__"], 60, 416, 608)
        _check_subset(d["train"], 60, 416, 608)
        _check_subset(d["test"], 50, 416, 608)

    for m in ("expert1_512", "expert2_512"):
        d = importlib.import_module(
            f"deepdraw.configs.datasets.drionsdb.{m}", package=__name__
        ).dataset
        assert len(d) == 4
        _check_subset(d["__train__"], 60, 512, 512)
        _check_subset(d["train"], 60, 512, 512)
        _check_subset(d["test"], 50, 512, 512)

    for m in ("expert1_768", "expert2_768"):
        d = importlib.import_module(
            f"deepdraw.configs.datasets.drionsdb.{m}", package=__name__
        ).dataset
        assert len(d) == 4
        _check_subset(d["__train__"], 60, 768, 768)
        _check_subset(d["train"], 60, 768, 768)
        _check_subset(d["test"], 50, 768, 768)


@pytest.mark.skip_if_rc_var_not_set("datadir.drive")
@pytest.mark.skip_if_rc_var_not_set("datadir.chasedb1")
@pytest.mark.skip_if_rc_var_not_set("datadir.hrf")
@pytest.mark.skip_if_rc_var_not_set("datadir.iostar")
def test_combined_vessels():
    def _check_subset(samples, size, height, width):
        assert len(samples) == size
        for s in samples[:N]:
            assert len(s) == 4
            assert isinstance(s[0], str)
            assert s[1].shape, (3, height == width)  # planes, height, width
            assert s[1].dtype == torch.float32
            assert s[2].shape, (1, height == width)  # planes, height, width
            assert s[2].dtype == torch.float32
            assert s[1].max() <= 1.0
            assert s[1].min() >= 0.0

    from deepdraw.configs.datasets.combined.vessel import dataset

    assert len(dataset) == 4
    _check_subset(dataset["__train__"], 73, 768, 768)
    _check_subset(dataset["__valid__"], 73, 768, 768)
    _check_subset(dataset["train"], 73, 768, 768)
    _check_subset(dataset["test"], 90, 768, 768)
