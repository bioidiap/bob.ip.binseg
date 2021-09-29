#!/usr/bin/env python
# coding=utf-8


"""Tests for Montgomery County CXR dataset"""

import numpy
import pytest

from ..data.montgomery import dataset
from .utils import count_bw


def test_protocol_consistency():

    subset = dataset.subsets("default")
    assert len(subset) == 3

    assert "train" in subset
    assert len(subset["train"]) == 96
    for s in subset["train"]:
        assert s.key.startswith("MontgomerySet")

    assert "validation" in subset
    assert len(subset["validation"]) == 14
    for s in subset["validation"]:
        assert s.key.startswith("MontgomerySet")

    assert "test" in subset
    assert len(subset["test"]) == 28
    for s in subset["test"]:
        assert s.key.startswith("MontgomerySet")


@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.montgomery.datadir")
def test_loading():

    image_size_1 = (4892, 4020)
    image_size_2 = (4020, 4892)

    def _check_sample(s, bw_threshold_label):

        data = s.data
        assert isinstance(data, dict)
        assert len(data) == 2

        assert "data" in data
        assert (
            data["data"].size == image_size_1
            or data["data"].size == image_size_2
        )
        assert data["data"].mode == "RGB"

        assert "label" in data
        assert (
            data["label"].size == image_size_1
            or data["label"].size == image_size_2
        )
        assert data["label"].mode == "1"

        b, w = count_bw(data["label"])
        assert (b + w) == numpy.prod(image_size_1), (
            f"Counts of black + white ({b}+{w}) do not add up to total "
            f"image size ({numpy.prod(image_size_1)}) at '{s.key}':label"
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
    proportions = [_check_sample(s, 0.67) for s in subset["train"][:limit]]
    proportions = [_check_sample(s, 0.67) for s in subset["validation"][:limit]]
    proportions = [_check_sample(s, 0.67) for s in subset["test"][:limit]]
    del proportions  # only to satisfy flake8


@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.montgomery.datadir")
def test_check():
    assert dataset.check() == 0
