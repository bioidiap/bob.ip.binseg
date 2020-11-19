#!/usr/bin/env python
# coding=utf-8


"""Tests for RIM-ONE r3"""

import os
import numpy
import pytest

from ..data.rimoner3 import dataset
from .utils import count_bw


def test_protocol_consistency():

    for protocol in ("optic-disc-exp1", "optic-cup-exp1", "optic-disc-exp2",
            "optic-cup-exp2", "optic-disc-avg", "optic-cup-avg"):

        subset = dataset.subsets(protocol)
        assert len(subset) == 2

        assert "train" in subset
        assert len(subset["train"]) == 99
        for s in subset["train"]:
            assert "Stereo Images" in s.key

        assert "test" in subset
        assert len(subset["test"]) == 60
        for s in subset["test"]:
            assert "Stereo Images" in s.key


@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.rimoner3.datadir")
@pytest.mark.slow
def test_loading():

    image_size = (1072, 1424)

    def _check_sample(s, bw_threshold_label):

        data = s.data
        assert isinstance(data, dict)
        assert len(data) == 2

        assert "data" in data
        assert data["data"].size == image_size
        assert data["data"].mode == "RGB"

        assert "label" in data
        assert data["label"].size == image_size
        assert data["label"].mode == "1"
        b, w = count_bw(data["label"])
        assert (b+w) == numpy.prod(image_size), \
                f"Counts of black + white ({b}+{w}) do not add up to total " \
                f"image size ({numpy.prod(image_size)}) at '{s.key}':label"
        assert (w/b) < bw_threshold_label, \
                f"The proportion between black and white pixels " \
                f"({w}/{b}={w/b:.2f}) is larger than the allowed threshold " \
                f"of {bw_threshold_label} at '{s.key}':label - this could " \
                f"indicate a loading problem!"

        # to visualize images, uncomment the folowing code
        # it should display an image with a faded background representing the
        # original data, blended with green labels.
        #from ..data.utils import overlayed_image
        #display = overlayed_image(data["data"], data["label"])
        #display.show()
        #import ipdb; ipdb.set_trace()

        return w/b

    subset = dataset.subsets("optic-cup-exp1")
    limit = None
    proportions = [_check_sample(s, 0.048) for s in subset["train"][:limit]]
    #print(f"max label proportions = {max(proportions)}")
    proportions = [_check_sample(s, 0.042) for s in subset["test"][:limit]]
    #print(f"max label proportions = {max(proportions)}")

    subset = dataset.subsets("optic-disc-exp1")
    proportions = [_check_sample(s, 0.088) for s in subset["train"][:limit]]
    #print(f"max label proportions = {max(proportions)}")
    proportions = [_check_sample(s, 0.061) for s in subset["test"][:limit]]
    #print(f"max label proportions = {max(proportions)}")

    subset = dataset.subsets("optic-cup-exp2")
    proportions = [_check_sample(s, 0.039) for s in subset["train"][:limit]]
    #print(f"max label proportions = {max(proportions)}")
    proportions = [_check_sample(s, 0.038) for s in subset["test"][:limit]]
    #print(f"max label proportions = {max(proportions)}")

    subset = dataset.subsets("optic-disc-exp2")
    proportions = [_check_sample(s, 0.090) for s in subset["train"][:limit]]
    #print(f"max label proportions = {max(proportions)}")
    proportions = [_check_sample(s, 0.065) for s in subset["test"][:limit]]
    #print(f"max label proportions = {max(proportions)}")

    subset = dataset.subsets("optic-cup-avg")
    proportions = [_check_sample(s, 0.042) for s in subset["train"][:limit]]
    #print(f"max label proportions = {max(proportions)}")
    proportions = [_check_sample(s, 0.040) for s in subset["test"][:limit]]
    #print(f"max label proportions = {max(proportions)}")

    subset = dataset.subsets("optic-disc-avg")
    proportions = [_check_sample(s, 0.089) for s in subset["train"][:limit]]
    #print(f"max label proportions = {max(proportions)}")
    proportions = [_check_sample(s, 0.063) for s in subset["test"][:limit]]
    #print(f"max label proportions = {max(proportions)}")


@pytest.mark.skip_if_rc_var_not_set("bob.ip.binseg.rimoner3.datadir")
@pytest.mark.slow
def test_check():
    assert dataset.check() == 0
