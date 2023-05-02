# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for DRHAGIS."""

import numpy
import pytest

from . import count_bw


def test_protocol_consistency():
    from deepdraw.data.drhagis import dataset

    subset = dataset.subsets("default")
    assert len(subset) == 2

    assert "train" in subset
    assert len(subset["train"]) == 19
    for s in subset["train"]:
        assert s.key.startswith("Fundus_Images")

    assert "test" in subset
    assert len(subset["test"]) == 20
    for s in subset["test"]:
        assert s.key.startswith("Fundus_Images")


@pytest.mark.skip_if_rc_var_not_set("datadir.drhagis")
def test_loading():
    from deepdraw.data.drhagis import dataset

    def _check_sample(s, bw_threshold_label, bw_threshold_mask):
        data = s.data
        assert isinstance(data, dict)
        assert len(data) == 3

        assert "data" in data
        assert data["data"].mode == "RGB"
        assert data["data"].size[0] >= 2816, (
            f"Width ({data['data'].size[0]}) for {s.key} is smaller "
            f"than 2816 pixels"
        )
        assert data["data"].size[1] >= 1880, (
            f"Width ({data['data'].size[1]}) for {s.key} is smaller "
            f"than 1880 pixels"
        )

        assert "label" in data
        assert data["data"].size == data["label"].size
        assert data["label"].mode == "1"

        b, w = count_bw(data["label"])
        assert (b + w) == numpy.prod(data["data"].size), (
            f"Counts of black + white ({b}+{w}) do not add up to total "
            f"image size ({numpy.prod(data['data'].size)}) at '{s.key}':label"
        )

        assert (w / b) < bw_threshold_label, (
            f"The proportion between black and white pixels "
            f"({w}/{b}={w/b:.3f}) is larger than the allowed threshold "
            f"of {bw_threshold_label} at '{s.key}':label - this could "
            f"indicate a loading problem!"
        )

        bm, wm = count_bw(data["mask"])
        assert (bm + wm) == numpy.prod(data["data"].size), (
            f"Counts of black + white ({bm}+{wm}) do not add up to total "
            f"image size ({numpy.prod(data['data'].size)}) at '{s.key}':mask"
        )

        assert (wm / bm) > bw_threshold_mask, (
            f"The proportion between black and white pixels in masks "
            f"({wm}/{bm}={wm/bm:.2f}) is smaller than the allowed "
            f"threshold of {bw_threshold_mask} at '{s.key}':label - "
            f"this could indicate a loading problem!"
        )

        # to visualize images, uncomment the folowing code
        # it should display an image with a faded background representing the
        # original data, blended with green labels and blue area indicating the
        # parts to be masked out.
        # from ..data.utils import overlayed_image
        # display = overlayed_image(data["data"], data["label"], data["mask"])
        # display.show()
        # import ipdb; ipdb.set_trace()

        return w / b, wm / bm

    limit = None  # use this to limit testing to first images only
    subset = dataset.subsets("default")
    proportions = [_check_sample(s, 0.07, 0.8) for s in subset["train"][:limit]]
    proportions = [_check_sample(s, 0.08, 0.8) for s in subset["test"][:limit]]
    del proportions  # only to satisfy flake8


@pytest.mark.skip_if_rc_var_not_set("datadir.drhagis")
def test_check():
    from deepdraw.data.drhagis import dataset

    assert dataset.check() == 0
