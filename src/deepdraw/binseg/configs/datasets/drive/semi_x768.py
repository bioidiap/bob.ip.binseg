# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Take DRIVE as labeled data and x,y as unlabeled data and can be HRF, STARE,
IOSTAR or other datasets."""
from ..drive.default_768 import dataset as _drive
from ..hrf.default_768 import dataset as _y2

# from ..iostar.vessel_768 import dataset as _x
from ..stare.ah_768 import dataset as _y1

dataset = {
    "train": _drive["train"]
    + _drive["test"],  # labeled dataset in semi-supervised learning
    "test": _y2["test"],  # test dataset
    "__valid__": _drive["train"] + _drive["test"],
    "__unlabeled_train__": _y1[
        "train"
    ],  # unlabeled dataset in semi-supervised learning
}
dataset["__extra_valid__"] = [dataset["test"]]
