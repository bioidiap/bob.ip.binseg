from bob.ip.binseg.data.transforms import *
_transforms = Compose(
    [
        Resize(544),
        Crop(0, 12, 544, 544),
        RandomHFlip(),
        RandomVFlip(),
        RandomRotation(),
        ColorJitter(),
        ToTensor(),
    ]
)

from bob.ip.binseg.data.utils import DelayedSample2TorchDataset
from bob.ip.binseg.data.chasedb1 import dataset as chasedb1
dataset = DelayedSample2TorchDataset(chasedb1.subsets("default")["train"],
        transform=_transforms)
