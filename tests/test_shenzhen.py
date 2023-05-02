# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for Shenzhen CXR dataset."""

import numpy
import pytest

from . import count_bw


def test_protocol_consistency():
    from deepdraw.data.shenzhen import dataset

    subset = dataset.subsets("default")
    assert len(subset) == 3

    assert "train" in subset
    assert len(subset["train"]) == 396
    for s in subset["train"]:
        assert s.key.startswith("CXR_png")

    assert "validation" in subset
    assert len(subset["validation"]) == 56
    for s in subset["validation"]:
        assert s.key.startswith("CXR_png")

    assert "test" in subset
    assert len(subset["test"]) == 114
    for s in subset["test"]:
        assert s.key.startswith("CXR_png")


@pytest.mark.skip_if_rc_var_not_set("datadir.shenzhen")
def test_loading():
    from deepdraw.data.shenzhen import dataset

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

        # to visualize images, uncomment the folowing code it should display an
        # image with a faded background representing the original data, blended
        # with green labels.
        # from ..data.utils import overlayed_image
        # display = overlayed_image(data["data"], data["label"])
        # display.show()
        # import ipdb; ipdb.set_trace()

        return w / b

    limit = None  # use this to limit testing to first images only
    subset = dataset.subsets("default")
    proportions = [_check_sample(s, 0.77) for s in subset["train"][:limit]]
    proportions = [_check_sample(s, 0.77) for s in subset["validation"][:limit]]
    proportions = [_check_sample(s, 0.77) for s in subset["test"][:limit]]
    del proportions  # only to satisfy flake8


@pytest.mark.skip_if_rc_var_not_set("datadir.shenzhen")
def test_check():
    from deepdraw.data.shenzhen import dataset

    assert dataset.check() == 0
