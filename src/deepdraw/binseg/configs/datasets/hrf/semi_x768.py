# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Take HRF as labeled data; x, y as unlabeled data and can be DRIVE, STARE,
IOSTAR or other datasets."""

from ..drive.default_768 import dataset as _y1
from ..hrf.default_768 import dataset as _hrf
from ..iostar.vessel_768 import dataset as _x
from ..stare.ah_768 import dataset as _y2

dataset = {
    "train": _hrf["train"]
    + _hrf["test"],  # labeled dataset in semi-supervised learning
    "test": _y1["test"],  # test dataset
    "__valid__": _hrf["train"] + _hrf["test"],
    "__unlabeled_train__": _y2[
        "train"
    ]  # unlabeled dataset in semi-supervised learning
    + _y2["test"]
    + _x["train"]
    + _x["test"],
}
dataset["__extra_valid__"] = [dataset["test"]]
