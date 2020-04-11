from bob.ip.binseg.data.transforms import *
_transforms = [
        Resize(544),
        Crop(0, 12, 544, 544),
        RandomHFlip(),
        RandomVFlip(),
        RandomRotation(),
        ColorJitter(),
        ]

from bob.ip.binseg.data.utils import SampleList2TorchDataset
from bob.ip.binseg.data.chasedb1 import dataset as chasedb1
dataset = SampleList2TorchDataset(chasedb1.subsets("default")["train"],
        transforms=_transforms)
