from bob.ip.binseg.data.transforms import *
_transforms = [
        RandomRotation(),
        Crop(140, 18, 680, 960),
        Resize(1168),
        RandomHFlip(),
        RandomVFlip(),
        ColorJitter(),
        ]

from bob.ip.binseg.data.utils import SampleList2TorchDataset
from bob.ip.binseg.data.chasedb1 import dataset as chasedb1
dataset = SampleList2TorchDataset(chasedb1.subsets("default")["train"],
        transforms=_transforms)
