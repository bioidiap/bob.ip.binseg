#!/usr/bin/env python
# coding=utf-8


"""Tests for DRIONS-DB"""

import os

import numpy
import nose.tools
from nose.plugins.attrib import attr

from ..data.drionsdb import dataset
from .utils import rc_variable_set, count_bw


def test_protocol_consistency():

    for protocol in ("expert1", "expert2"):

        subset = dataset.subsets(protocol)
        nose.tools.eq_(len(subset), 2)

        assert "train" in subset
        nose.tools.eq_(len(subset["train"]), 60)
        for s in subset["train"]:
            assert s.key.startswith(os.path.join("images", "image_0"))

        assert "test" in subset
        nose.tools.eq_(len(subset["test"]), 50)
        for s in subset["test"]:
            assert s.key.startswith(os.path.join("images", "image_"))


@rc_variable_set("bob.ip.binseg.drionsdb.datadir")
@attr("slow")
def test_loading():

    image_size = (600, 400)

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
        assert (b + w) == numpy.prod(image_size), (
            f"Counts of black + white ({b}+{w}) do not add up to total "
            f"image size ({numpy.prod(image_size)}) at '{s.key}':label"
        )
        assert (w / b) < bw_threshold_label, (
            f"The proportion between black and white pixels "
            f"({w}/{b}={w/b:.3f}) is larger than the allowed threshold "
            f"of {bw_threshold_label} at '{s.key}':label - this could "
            f"indicate a loading problem!"
        )

        # to visualize images, uncomment the folowing code
        # it should display an image with a faded background representing the
        # original data, blended with green labels.
        #from ..data.utils import overlayed_image
        #display = overlayed_image(data["data"], data["label"])
        #display.show()
        #import ipdb; ipdb.set_trace()

        return w/b

    limit = None  #use this to limit testing to first images only
    subset = dataset.subsets("expert1")
    proportions = [_check_sample(s, 0.046) for s in subset["train"][:limit]]
    #print(f"max label proportions = {max(proportions)}")
    proportions = [_check_sample(s, 0.043) for s in subset["test"][:limit]]
    #print(f"max label proportions = {max(proportions)}")

    subset = dataset.subsets("expert2")
    proportions = [_check_sample(s, 0.044) for s in subset["train"][:limit]]
    #print(f"max label proportions = {max(proportions)}")
    proportions = [_check_sample(s, 0.045) for s in subset["test"][:limit]]
    #print(f"max label proportions = {max(proportions)}")

@rc_variable_set("bob.ip.binseg.drionsdb.datadir")
@attr("slow")
def test_check():
    nose.tools.eq_(dataset.check(), 0)
