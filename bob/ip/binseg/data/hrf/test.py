#!/usr/bin/env python
# coding=utf-8


"""Tests for HRF"""

import os
import nose.tools

from ..utils import rc_variable_set, DelayedSample2TorchDataset
from ..transforms import Compose, Crop, Resize
from . import dataset


def test_protocol_consitency():

    subset = dataset.subsets("default")
    nose.tools.eq_(len(subset), 2)

    assert "train" in subset
    nose.tools.eq_(len(subset["train"]), 15)
    for s in subset["train"]:
        assert s.key.startswith(os.path.join("images", "0"))

    assert "test" in subset
    nose.tools.eq_(len(subset["test"]), 30)
    for s in subset["test"]:
        assert s.key.startswith("images")


@rc_variable_set('bob.ip.binseg.hrf.datadir')
def test_loading():

    def _check_sample(s):
        data = s.data
        assert isinstance(data, dict)
        nose.tools.eq_(len(data), 3)
        assert "data" in data
        nose.tools.eq_(data["data"].size, (3504, 2336))
        nose.tools.eq_(data["data"].mode, "RGB")
        assert "label" in data
        nose.tools.eq_(data["label"].size, (3504, 2336))
        nose.tools.eq_(data["label"].mode, "1")
        assert "mask" in data
        nose.tools.eq_(data["mask"].size, (3504, 2336))
        nose.tools.eq_(data["mask"].mode, "1")

    subset = dataset.subsets("default")
    for s in subset["train"]: _check_sample(s)
    for s in subset["test"]: _check_sample(s)


@rc_variable_set('bob.ip.binseg.hrf.datadir')
def test_check():
    nose.tools.eq_(dataset.check(), 0)


@rc_variable_set('bob.ip.binseg.hrf.datadir')
def test_torch_dataset():

    def _check_sample(s):
        nose.tools.eq_(len(s), 4)
        assert isinstance(s[0], str)
        nose.tools.eq_(s[1].size, (1648, 1168))
        nose.tools.eq_(s[1].mode, "RGB")
        nose.tools.eq_(s[2].size, (1648, 1168))
        nose.tools.eq_(s[2].mode, "1")
        nose.tools.eq_(s[3].size, (1648, 1168))
        nose.tools.eq_(s[3].mode, "1")


    transforms = Compose([Crop(0, 108, 2336, 3296), Resize((1168))])

    subset = dataset.subsets("default")

    torch_dataset = DelayedSample2TorchDataset(subset["train"], transforms)
    nose.tools.eq_(len(torch_dataset), 15)
    for s in torch_dataset: _check_sample(s)

    torch_dataset = DelayedSample2TorchDataset(subset["test"], transforms)
    nose.tools.eq_(len(torch_dataset), 30)
    for s in torch_dataset: _check_sample(s)
