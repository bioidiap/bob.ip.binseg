# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for Japanese Society of Radiological Technology."""

import numpy
import pytest

from . import count_bw


def test_protocol_consistency():
    from deepdraw.data.jsrt import dataset

    subset = dataset.subsets("default")
    assert len(subset) == 3

    assert "train" in subset
    assert len(subset["train"]) == 172
    for s in subset["train"]:
        assert s.key.startswith("All247images")

    assert "validation" in subset
    assert len(subset["validation"]) == 25
    for s in subset["validation"]:
        assert s.key.startswith("All247images")

    assert "test" in subset
    assert len(subset["test"]) == 50
    for s in subset["test"]:
        assert s.key.startswith("All247images")


@pytest.mark.skip_if_rc_var_not_set("datadir.jsrt")
def test_loading():
    from deepdraw.data.jsrt import dataset

    image_size = (1024, 1024)

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
    proportions = [_check_sample(s, 0.85) for s in subset["train"][:limit]]
    proportions = [_check_sample(s, 0.85) for s in subset["validation"][:limit]]
    proportions = [_check_sample(s, 0.85) for s in subset["test"][:limit]]
    del proportions  # only to satisfy flake8


@pytest.mark.skip_if_rc_var_not_set("datadir.jsrt")
def test_check():
    from deepdraw.data.jsrt import dataset

    assert dataset.check() == 0
