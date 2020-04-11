from bob.ip.binseg.data.transforms import *
_transforms = [
        RandomRotation(),
        CenterCrop((829, 960)),
        Resize(608),
        RandomHFlip(),
        RandomVFlip(),
        ColorJitter(),
        ]

from bob.ip.binseg.data.utils import SampleList2TorchDataset
from bob.ip.binseg.data.chasedb1 import dataset as chasedb1
dataset = SampleList2TorchDataset(chasedb1.subsets("default")["train"],
        transforms=_transforms)
