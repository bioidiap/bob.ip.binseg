#!/usr/bin/env python
# coding=utf-8


"""Tests for DRIVE"""

import os
import numpy
import pytest

from ..data.drive import dataset
from .utils import count_bw


def test_protocol_consistency():

    subset = dataset.subsets("default")
    assert len(subset) == 2

    assert "train" in subset
    assert len(subset["train"]) == 20
    for s in subset["train"]:
        assert s.key.startswith(os.path.join("training", "images"))

    assert "test" in subset
    assert len(subset["test"]) == 20
    for s in subset["test"]:
        assert s.key.startswith(os.path.join("test", "images"))

    subset = dataset.subsets("second-annotator")
    assert len(subset) == 1

    assert "test" in subset
    assert len(subset["test"]) == 20
    for s in subset["test"]:
        assert s.key.startswith(os.path.join("test", "images"))


@pytest.mark.skip_if_rc_var_not_set('bob.ip.binseg.drive.datadir')
def test_loading():

    image_size = (565, 584)

    def _check_sample(s, bw_threshold_label, bw_threshold_mask):

        data = s.data
        assert isinstance(data, dict)
        assert len(data) == 3

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
                f"The proportion between black and white pixels in labels " \
                f"({w}/{b}={w/b:.2f}) is larger than the allowed threshold " \
                f"of {bw_threshold_label} at '{s.key}':label - this could " \
                f"indicate a loading problem!"

        assert "mask" in data
        assert data["mask"].size == image_size
        assert data["mask"].mode == "1"
        bm, wm = count_bw(data["mask"])
        assert (bm+wm) == numpy.prod(image_size), \
                f"Counts of black + white ({bm}+{wm}) do not add up to total " \
                f"image size ({numpy.prod(image_size)}) at '{s.key}':mask"
        assert (wm/bm) > bw_threshold_mask, \
                f"The proportion between black and white pixels in masks " \
                f"({wm}/{bm}={wm/bm:.2f}) is smaller than the allowed " \
                f"threshold of {bw_threshold_mask} at '{s.key}':label - " \
                f"this could indicate a loading problem!"

        # to visualize images, uncomment the folowing code
        # it should display an image with a faded background representing the
        # original data, blended with green labels and blue area indicating the
        # parts to be masked out.
        #from ..data.utils import overlayed_image
        #display = overlayed_image(data["data"], data["label"], data["mask"])
        #display.show()
        #import ipdb; ipdb.set_trace()

        return w/b, wm/bm

    limit = None  #use this to limit testing to first images only
    subset = dataset.subsets("default")
    proportions = [_check_sample(s, 0.14, 2.14) for s in subset["train"][:limit]]
    #print(f"max label proportions = {max(k[0] for k in proportions)}")
    #print(f"min mask proportions = {min(k[1] for k in proportions)}")
    proportions = [_check_sample(s, 0.12, 2.12) for s in subset["test"]][:limit]
    #print(f"max label proportions = {max(k[0] for k in proportions)}")
    #print(f"min mask proportions = {min(k[1] for k in proportions)}")

    subset = dataset.subsets("second-annotator")
    proportions = [_check_sample(s, 0.12, 2.12) for s in subset["test"][:limit]]
    #print(f"max label proportions = {max(k[0] for k in proportions)}")
    #print(f"min mask proportions = {min(k[1] for k in proportions)}")


@pytest.mark.skip_if_rc_var_not_set('bob.ip.binseg.drive.datadir')
def test_check():
    assert dataset.check() == 0
