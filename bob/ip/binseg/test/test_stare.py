#!/usr/bin/env python
# coding=utf-8


"""Tests for STARE"""

import os

import numpy
import nose.tools

## special trick for CI builds
from . import mock_dataset
dataset, rc_variable_set = mock_dataset()

from .utils import count_bw


def test_protocol_consistency():

    subset = dataset.subsets("default")
    nose.tools.eq_(len(subset), 2)

    assert "train" in subset
    nose.tools.eq_(len(subset["train"]), 10)
    for s in subset["train"]:
        assert s.key.startswith(os.path.join("stare-images", "im0"))

    assert "test" in subset
    nose.tools.eq_(len(subset["test"]), 10)
    for s in subset["test"]:
        assert s.key.startswith(os.path.join("stare-images", "im0"))

    subset = dataset.subsets("second-annotation")
    nose.tools.eq_(len(subset), 2)

    assert "train" in subset
    nose.tools.eq_(len(subset["train"]), 10)
    for s in subset["train"]:
        assert s.key.startswith(os.path.join("stare-images", "im0"))

    assert "test" in subset
    nose.tools.eq_(len(subset["test"]), 10)
    for s in subset["test"]:
        assert s.key.startswith(os.path.join("stare-images", "im0"))


@rc_variable_set('bob.ip.binseg.stare.datadir')
def test_loading():

    image_size = (700, 605)

    def _check_sample(s, bw_threshold_label):

        data = s.data
        assert isinstance(data, dict)
        nose.tools.eq_(len(data), 2)

        assert "data" in data
        nose.tools.eq_(data["data"].size, image_size)
        nose.tools.eq_(data["data"].mode, "RGB")

        assert "label" in data
        nose.tools.eq_(data["label"].size, image_size)
        nose.tools.eq_(data["label"].mode, "1")
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

    limit = None  #use this to limit testing to first images only
    subset = dataset.subsets("default")
    proportions = [_check_sample(s, 0.10) for s in subset["train"][:limit]]
    #print(f"max label proportions = {max(proportions)}")
    proportions = [_check_sample(s, 0.12) for s in subset["test"][:limit]]
    #print(f"max label proportions = {max(proportions)}")

    subset = dataset.subsets("second-annotation")
    proportions = [_check_sample(s, 0.19) for s in subset["train"][:limit]]
    #print(f"max label proportions = {max(proportions)}")
    proportions = [_check_sample(s, 0.18) for s in subset["test"][:limit]]
    #print(f"max label proportions = {max(proportions)}")


@rc_variable_set('bob.ip.binseg.stare.datadir')
def test_check():
    nose.tools.eq_(dataset.check(), 0)
