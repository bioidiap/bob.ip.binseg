from bob.ip.binseg.data.transforms import *
_transforms = Compose(
    [
        RandomRotation(),
        Crop(0, 18, 960, 960),
        Resize(1024),
        RandomHFlip(),
        RandomVFlip(),
        ColorJitter(),
        ToTensor(),
    ]
)

from bob.ip.binseg.data.utils import DelayedSample2TorchDataset
from bob.ip.binseg.data.chasedb1 import dataset as chasedb1
dataset = DelayedSample2TorchDataset(chasedb1.subsets("default")["train"],
        transform=_transforms)
