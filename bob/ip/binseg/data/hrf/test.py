#!/usr/bin/env python
# coding=utf-8


"""Tests for HRF"""

import os

import numpy
import nose.tools

from . import dataset
from ...test.utils import rc_variable_set


def test_protocol_consistency():

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

    from ..utils import count_bw
    image_size = (3504, 2336)
    bw_threshold_label = 0.12  #(vessels to background proportion limit)

    def _check_sample(s):

        data = s.data
        assert isinstance(data, dict)
        nose.tools.eq_(len(data), 3)

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

        assert "mask" in data
        nose.tools.eq_(data["mask"].size, image_size)
        nose.tools.eq_(data["mask"].mode, "1")
        b, w = count_bw(data["mask"])
        assert (b+w) == numpy.prod(image_size), \
                f"Counts of black + white ({b}+{w}) do not add up to total " \
                f"image size ({numpy.prod(image_size)}) at '{s.key}':mask"
        assert w > b, \
                f"The proportion between white and black pixels " \
                f"({w} > {b}?) is not respected at '{s.key}':mask - " \
                f"this could indicate a loading problem!"

        # to visualize images, uncomment the folowing code
        # it should display an image with a faded background representing the
        # original data, blended with green labels and blue area indicating the
        # parts to be masked out.
        #from ..utils import overlayed_image
        #display = overlayed_image(data["data"], data["label"], data["mask"])
        #display.show()
        #import ipdb; ipdb.set_trace()
        #pass

    subset = dataset.subsets("default")
    for s in subset["train"]: _check_sample(s)
    for s in subset["test"]: _check_sample(s)


@rc_variable_set('bob.ip.binseg.hrf.datadir')
def test_check():
    nose.tools.eq_(dataset.check(), 0)
