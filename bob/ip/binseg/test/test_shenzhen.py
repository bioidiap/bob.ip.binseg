#!/usr/bin/env python
# coding=utf-8


"""Tests for Shenzhen CXR dataset"""

import numpy
import pytest

from ..data.shenzhen import dataset
from .utils import count_bw


def test_protocol_consistency():

    subset = dataset.subsets("default")
    assert len(subset) == 3

    assert "train" in subset
    assert len(subset["train"]) == 396
    for s in subset["train"]:
        assert s.key.startswith("ChinaSet_AllFiles")

    assert "validation" in subset
    assert len(subset["validation"]) == 56
    for s in subset["validation"]:
        assert s.key.startswith("ChinaSet_AllFiles")

    assert "test" in subset
    assert len(subset["test"]) == 114
    for s in subset["test"]:
        assert s.key.startswith("ChinaSet_AllFiles")


@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.shenzhen.datadir")
def test_loading():

    min_image_size = (1130, 948)
    max_image_size = (3001, 3001)

    def _check_sample(s, bw_threshold_label):

        data = s.data
        assert isinstance(data, dict)
        assert len(data) == 2

        assert "data" in data
        assert data["data"].mode == "RGB"

        assert "label" in data
        assert data["label"].mode == "1"

        b, w = count_bw(data["label"])
        assert (b + w) >= numpy.prod(min_image_size), (
            f"Counts of black + white ({b}+{w}) lower than smallest image total"
            f"image size ({numpy.prod(min_image_size)}) at '{s.key}':label"
        )
        assert (b + w) <= numpy.prod(max_image_size), (
            f"Counts of black + white ({b}+{w}) higher than largest image total"
            f"image size ({numpy.prod(max_image_size)}) at '{s.key}':label"
        )
        assert (w / b) < bw_threshold_label, (
            f"The proportion between black and white pixels "
            f"({w}/{b}={w/b:.3f}) is larger than the allowed threshold "
            f"of {bw_threshold_label} at '{s.key}':label - this could "
            f"indicate a loading problem!"
        )

        return w / b

    limit = None  # use this to limit testing to first images only
    subset = dataset.subsets("default")
    proportions = [_check_sample(s, 0.77) for s in subset["train"][:limit]]
    proportions = [_check_sample(s, 0.77) for s in subset["validation"][:limit]]
    proportions = [_check_sample(s, 0.77) for s in subset["test"][:limit]]
    del proportions  # only to satisfy flake8


@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.shenzhen.datadir")
def test_check():
    assert dataset.check() == 0
