# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for STARE."""

import os

import numpy
import pytest

from . import count_bw


def test_protocol_consistency():
    from deepdraw.data.stare import dataset

    subset = dataset.subsets("ah")
    assert len(subset) == 2

    assert "train" in subset
    assert len(subset["train"]) == 10
    for s in subset["train"]:
        assert s.key.startswith(os.path.join("stare-images", "im0"))

    assert "test" in subset
    assert len(subset["test"]) == 10
    for s in subset["test"]:
        assert s.key.startswith(os.path.join("stare-images", "im0"))

    subset = dataset.subsets("vk")
    assert len(subset) == 2

    assert "train" in subset
    assert len(subset["train"]) == 10
    for s in subset["train"]:
        assert s.key.startswith(os.path.join("stare-images", "im0"))

    assert "test" in subset
    assert len(subset["test"]) == 10
    for s in subset["test"]:
        assert s.key.startswith(os.path.join("stare-images", "im0"))


@pytest.mark.skip_if_rc_var_not_set("datadir.stare")
def test_loading():
    from deepdraw.data.stare import dataset

    image_size = (700, 605)

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
        assert (b + w) == numpy.prod(image_size), (
            f"Counts of black + white ({b}+{w}) do not add up to total "
            f"image size ({numpy.prod(image_size)}) at '{s.key}':label"
        )
        assert (w / b) < bw_threshold_label, (
            f"The proportion between black and white pixels "
            f"({w}/{b}={w/b:.2f}) is larger than the allowed threshold "
            f"of {bw_threshold_label} at '{s.key}':label - this could "
            f"indicate a loading problem!"
        )

        assert "mask" in data
        assert data["mask"].size == image_size
        assert data["mask"].mode == "1"
        bm, wm = count_bw(data["mask"])
        assert (bm + wm) == numpy.prod(image_size), (
            f"Counts of black + white ({bm}+{wm}) do not add up to total "
            f"image size ({numpy.prod(image_size)}) at '{s.key}':mask"
        )
        assert (wm / bm) > bw_threshold_mask, (
            f"The proportion between black and white pixels in masks "
            f"({wm}/{bm}={wm/bm:.2f}) is smaller than the allowed "
            f"threshold of {bw_threshold_mask} at '{s.key}':label - "
            f"this could indicate a loading problem!"
        )
        # print (f"{s.key}: {wm/bm} > {bw_threshold_mask}? {(wm/bm)>bw_threshold_mask}")

        # to visualize images, uncomment the folowing code
        # it should display an image with a faded background representing the
        # original data, blended with green labels.
        # from ..data.utils import overlayed_image
        # display = overlayed_image(data["data"], data["label"], data["mask"])
        # display.show()
        # import ipdb; ipdb.set_trace()

        return w / b

    limit = None  # use this to limit testing to first images only
    subset = dataset.subsets("ah")
    proportions = [
        _check_sample(s, 0.10, 2.67) for s in subset["train"][:limit]
    ]
    # print(f"max label proportions = {max(proportions)}")
    proportions = [_check_sample(s, 0.12, 2.70) for s in subset["test"][:limit]]
    # print(f"max label proportions = {max(proportions)}")

    subset = dataset.subsets("vk")
    # proportions = [_check_sample(s, 0.19, 2.67) for s in subset["train"][:limit]]
    # print(f"max label proportions = {max(proportions)}")
    # proportions = [_check_sample(s, 0.18, 2.70) for s in subset["test"][:limit]]
    # print(f"max label proportions = {max(proportions)}")
    del proportions  # only to satisfy flake8


@pytest.mark.skip_if_rc_var_not_set("datadir.stare")
def test_check():
    from deepdraw.data.stare import dataset

    assert dataset.check() == 0
