from bob.ip.binseg.data.transforms import Resize, Crop
from bob.ip.binseg.configs.datasets.utils import DATA_AUGMENTATION as _DA
_transforms = [Resize(544), Crop(0, 12, 544, 544)] + _DA

from bob.ip.binseg.data.utils import SampleList2TorchDataset
from bob.ip.binseg.data.chasedb1 import dataset as chasedb1
dataset = SampleList2TorchDataset(chasedb1.subsets("default")["train"],
        transforms=_transforms)
