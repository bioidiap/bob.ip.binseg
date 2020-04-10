from bob.ip.binseg.data.transforms import *
_transforms = Compose(
    [
        RandomRotation(),
        CenterCrop((829, 960)),
        Resize(608),
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
