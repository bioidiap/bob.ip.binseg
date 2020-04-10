#!/usr/bin/env python
# coding=utf-8


"""Tests for IOSTAR"""

import os
import nose.tools

from ..utils import rc_variable_set, DelayedSample2TorchDataset
from ..transforms import Compose, Crop, Resize
from . import dataset


def test_protocol_consitency():

    subset = dataset.subsets("vessel")
    nose.tools.eq_(len(subset), 2)

    assert "train" in subset
    nose.tools.eq_(len(subset["train"]), 20)
    for s in subset["train"]:
        assert s.key.startswith(os.path.join("image", "STAR "))

    assert "test" in subset
    nose.tools.eq_(len(subset["test"]), 10)
    for s in subset["test"]:
        assert s.key.startswith(os.path.join("image", "STAR "))

    subset = dataset.subsets("optic-disc")
    nose.tools.eq_(len(subset), 2)

    assert "train" in subset
    nose.tools.eq_(len(subset["train"]), 20)
    for s in subset["train"]:
        assert s.key.startswith(os.path.join("image", "STAR "))

    assert "test" in subset
    nose.tools.eq_(len(subset["test"]), 10)
    for s in subset["test"]:
        assert s.key.startswith(os.path.join("image", "STAR "))


@rc_variable_set('bob.ip.binseg.iostar.datadir')
def test_loading():

    def _check_sample(s):
        data = s.data
        assert isinstance(data, dict)
        nose.tools.eq_(len(data), 3)
        assert "data" in data
        nose.tools.eq_(data["data"].size, (1024, 1024))
        nose.tools.eq_(data["data"].mode, "RGB")
        assert "label" in data
        nose.tools.eq_(data["label"].size, (1024, 1024))
        nose.tools.eq_(data["label"].mode, "1")
        assert "label" in data
        nose.tools.eq_(data["label"].size, (1024, 1024))
        nose.tools.eq_(data["label"].mode, "1")

    subset = dataset.subsets("vessel")
    for s in subset["train"]: _check_sample(s)
    for s in subset["test"]: _check_sample(s)

    subset = dataset.subsets("optic-disc")
    for s in subset["train"]: _check_sample(s)
    for s in subset["test"]: _check_sample(s)


@rc_variable_set('bob.ip.binseg.iostar.datadir')
def test_check():
    nose.tools.eq_(dataset.check(), 0)


@rc_variable_set('bob.ip.binseg.stare.datadir')
def test_torch_dataset():

    def _check_sample(s):
        nose.tools.eq_(len(s), 4)
        assert isinstance(s[0], str)
        nose.tools.eq_(s[1].size, (1024, 1024))
        nose.tools.eq_(s[1].mode, "RGB")
        nose.tools.eq_(s[2].size, (1024, 1024))
        nose.tools.eq_(s[2].mode, "1")
        nose.tools.eq_(s[3].size, (1024, 1024))
        nose.tools.eq_(s[3].mode, "1")

    transforms = None

    subset = dataset.subsets("vessel")

    torch_dataset = DelayedSample2TorchDataset(subset["train"], transforms)
    nose.tools.eq_(len(torch_dataset), 20)
    for s in torch_dataset: _check_sample(s)

    torch_dataset = DelayedSample2TorchDataset(subset["test"], transforms)
    nose.tools.eq_(len(torch_dataset), 10)
    for s in torch_dataset: _check_sample(s)

    subset = dataset.subsets("optic-disc")

    torch_dataset = DelayedSample2TorchDataset(subset["train"], transforms)
    nose.tools.eq_(len(torch_dataset), 20)
    for s in torch_dataset: _check_sample(s)

    torch_dataset = DelayedSample2TorchDataset(subset["test"], transforms)
    nose.tools.eq_(len(torch_dataset), 10)
    for s in torch_dataset: _check_sample(s)
