#!/usr/bin/env python
# coding=utf-8


"""Tests for STARE"""

import os

import numpy
import nose.tools

from . import dataset
from ...test.utils import rc_variable_set


def test_protocol_consitency():

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

    from ..utils import count_bw
    image_size = (700, 605)
    bw_threshold_label = 0.19  #(vessels to background proportion limit)

    def _check_sample(s):

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
        #from ..utils import overlayed_image
        #display = overlayed_image(data["data"], data["label"])
        #display.show()
        #import ipdb; ipdb.set_trace()
        #pass

    subset = dataset.subsets("default")
    for s in subset["train"]: _check_sample(s)
    for s in subset["test"]: _check_sample(s)

    subset = dataset.subsets("second-annotation")
    for s in subset["test"]: _check_sample(s)


@rc_variable_set('bob.ip.binseg.stare.datadir')
def test_check():
    nose.tools.eq_(dataset.check(), 0)
