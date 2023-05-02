# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for Drishti-GS1."""

import os

import numpy
import pytest

from . import count_bw


def test_protocol_consistency():
    from deepdraw.data.drishtigs1 import dataset

    for protocol in (
        "optic-disc-all",
        "optic-cup-all",
        "optic-disc-any",
        "optic-cup-any",
    ):
        subset = dataset.subsets(protocol)
        assert len(subset) == 2

        assert "train" in subset
        assert len(subset["train"]) == 50
        for s in subset["train"]:
            assert s.key.startswith(
                os.path.join(
                    "Drishti-GS1_files", "Training", "Images", "drishtiGS_"
                )
            )

        assert "test" in subset
        assert len(subset["test"]) == 51
        for s in subset["test"]:
            assert s.key.startswith(
                os.path.join(
                    "Drishti-GS1_files", "Test", "Images", "drishtiGS_"
                )
            )


@pytest.mark.skip_if_rc_var_not_set("datadir.drishtigs1")
@pytest.mark.slow
def test_loading():
    from deepdraw.data.drishtigs1 import dataset

    def _check_sample(s, bw_threshold_label):
        data = s.data
        assert isinstance(data, dict)
        assert len(data) == 3

        assert "data" in data
        assert data["data"].size[0] > 2040, (
            f"Width ({data['data'].size[0]}) for {s.key} is smaller "
            f"than 2040 pixels"
        )
        assert data["data"].size[1] > 1740, (
            f"Width ({data['data'].size[1]}) for {s.key} is smaller "
            f"than 1740 pixels"
        )
        assert data["data"].mode == "RGB"

        assert "label" in data
        # assert data["label"].size == image_size
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

        assert "mask" in data
        assert data["data"].size == data["mask"].size
        assert data["mask"].mode == "1"

        # to visualize images, uncomment the folowing code
        # it should display an image with a faded background representing the
        # original data, blended with green labels.
        # from ..data.utils import overlayed_image
        # display = overlayed_image(data["data"], data["label"])
        # display.show()
        # import ipdb; ipdb.set_trace()

        return w / b

    limit = None
    subset = dataset.subsets("optic-cup-all")
    proportions = [_check_sample(s, 0.027) for s in subset["train"][:limit]]
    # print(f"max label proportions = {max(proportions)}")
    proportions = [_check_sample(s, 0.035) for s in subset["test"][:limit]]
    # print(f"max label proportions = {max(proportions)}")

    subset = dataset.subsets("optic-disc-all")
    proportions = [_check_sample(s, 0.045) for s in subset["train"][:limit]]
    # print(f"max label proportions = {max(proportions)}")
    proportions = [_check_sample(s, 0.055) for s in subset["test"][:limit]]
    # print(f"max label proportions = {max(proportions)}")

    subset = dataset.subsets("optic-cup-any")
    proportions = [_check_sample(s, 0.034) for s in subset["train"][:limit]]
    # print(f"max label proportions = {max(proportions)}")
    proportions = [_check_sample(s, 0.047) for s in subset["test"][:limit]]
    # print(f"max label proportions = {max(proportions)}")

    subset = dataset.subsets("optic-disc-any")
    proportions = [_check_sample(s, 0.052) for s in subset["train"][:limit]]
    # print(f"max label proportions = {max(proportions)}")
    proportions = [_check_sample(s, 0.060) for s in subset["test"][:limit]]
    # print(f"max label proportions = {max(proportions)}")
    del proportions  # only to satisfy flake8


@pytest.mark.skip_if_rc_var_not_set("datadir.drishtigs1")
@pytest.mark.slow
def test_check():
    from deepdraw.data.drishtigs1 import dataset

    assert dataset.check() == 0
