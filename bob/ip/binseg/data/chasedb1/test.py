#!/usr/bin/env python
# coding=utf-8


"""Tests for CHASE-DB1"""

import os
import nose.tools

from . import dataset
from ...test.utils import rc_variable_set


def test_protocol_consitency():

    subset = dataset.subsets("default")
    nose.tools.eq_(len(subset), 2)

    assert "train" in subset
    nose.tools.eq_(len(subset["train"]), 8)
    for s in subset["train"]:
        assert s.key.startswith("Image_")

    assert "test" in subset
    nose.tools.eq_(len(subset["test"]), 20)
    for s in subset["test"]:
        assert s.key.startswith("Image_")

    subset = dataset.subsets("second-annotation")
    nose.tools.eq_(len(subset), 2)

    assert "train" in subset
    nose.tools.eq_(len(subset["train"]), 8)
    for s in subset["train"]:
        assert s.key.startswith("Image_")

    assert "test" in subset
    nose.tools.eq_(len(subset["test"]), 20)
    for s in subset["test"]:
        assert s.key.startswith("Image_")


@rc_variable_set('bob.ip.binseg.chasedb1.datadir')
def test_loading():

    def _check_sample(s):
        data = s.data
        assert isinstance(data, dict)
        nose.tools.eq_(len(data), 2)
        assert "data" in data
        nose.tools.eq_(data["data"].size, (999, 960))
        nose.tools.eq_(data["data"].mode, "RGB")
        assert "label" in data
        nose.tools.eq_(data["label"].size, (999, 960))
        nose.tools.eq_(data["label"].mode, "1")

    subset = dataset.subsets("default")
    for s in subset["train"]: _check_sample(s)
    for s in subset["test"]: _check_sample(s)

    subset = dataset.subsets("second-annotation")
    for s in subset["test"]: _check_sample(s)


@rc_variable_set('bob.ip.binseg.chasedb1.datadir')
def test_check():
    nose.tools.eq_(dataset.check(), 0)
