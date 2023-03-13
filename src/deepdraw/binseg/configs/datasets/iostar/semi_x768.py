"""Take IOSTAR as labeled data and x as unlabeled data
    x can be DRIVE, HRF, STARE or other datasets
"""

from ..drive.default_768 import dataset as _x
from ..hrf.default_768 import dataset as _y2
from ..iostar.vessel_768 import dataset as _iostar
from ..stare.ah_768 import dataset as _y1

dataset = {
    "train": _iostar["train"]
    + _iostar["test"],  # labeled dataset in semi-supervised learning
    "test": _y2["test"],  # test dataset
    "__valid__": _iostar["train"] + _iostar["test"],
    "__unlabeled_train__": _y1["train"]
    + _y1["test"]
    + _x["train"]
    + _x["test"],  # unlabeled dataset in semi-supervised learning
}
dataset["__extra_valid__"] = [dataset["test"]]
