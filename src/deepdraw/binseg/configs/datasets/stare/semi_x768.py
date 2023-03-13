"""Take STARE as labeled data and X as unlabeled data
    X can be  DRIVE, HRF, IOSTAR or other datasets
"""

from ..drive.default_768 import dataset as _y1
from ..hrf.default_768 import dataset as _y2
from ..iostar.vessel_768 import dataset as _x
from ..stare.ah_768 import dataset as _stare

dataset = {
    "train": _stare["train"]
    + _stare["test"],  # labeled dataset in semi-supervised learning
    "test": _x["test"],  # test dataset
    "__valid__": _stare["train"] + _stare["test"],
    "__unlabeled_train__": _y1["train"]
    + _y1["test"],  # unlabeled dataset in semi-supervised learning
}
dataset["__extra_valid__"] = [dataset["test"]]
