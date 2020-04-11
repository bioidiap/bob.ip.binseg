#!/usr/bin/env python
# coding=utf-8


"""Tests for DRIVE"""

import os
import nose.tools

from . import dataset
from ...test.utils import rc_variable_set


def test_protocol_consitency():

    subset = dataset.subsets("default")
    nose.tools.eq_(len(subset), 2)

    assert "train" in subset
    nose.tools.eq_(len(subset["train"]), 20)
    for s in subset["train"]:
        assert s.key.startswith(os.path.join("training", "images"))

    assert "test" in subset
    nose.tools.eq_(len(subset["test"]), 20)
    for s in subset["test"]:
        assert s.key.startswith(os.path.join("test", "images"))

    subset = dataset.subsets("second-annotation")
    nose.tools.eq_(len(subset), 1)

    assert "test" in subset
    nose.tools.eq_(len(subset["test"]), 20)
    for s in subset["test"]:
        assert s.key.startswith(os.path.join("test", "images"))


@rc_variable_set('bob.ip.binseg.drive.datadir')
def test_loading():

    def _check_sample(s):
        data = s.data
        assert isinstance(data, dict)
        nose.tools.eq_(len(data), 3)
        assert "data" in data
        nose.tools.eq_(data["data"].size, (565, 584))
        nose.tools.eq_(data["data"].mode, "RGB")
        assert "label" in data
        nose.tools.eq_(data["label"].size, (565, 584))
        nose.tools.eq_(data["label"].mode, "1")
        assert "mask" in data
        nose.tools.eq_(data["mask"].size, (565, 584))
        nose.tools.eq_(data["mask"].mode, "1")

    subset = dataset.subsets("default")
    for s in subset["train"]: _check_sample(s)
    for s in subset["test"]: _check_sample(s)

    subset = dataset.subsets("second-annotation")
    for s in subset["test"]: _check_sample(s)


@rc_variable_set('bob.ip.binseg.drive.datadir')
def test_check():
    nose.tools.eq_(dataset.check(), 0)
