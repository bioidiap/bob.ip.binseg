from bob.ip.binseg.configs.datasets.drive1024 import dataset as drive
from bob.ip.binseg.configs.datasets.stare1024 import dataset as stare
from bob.ip.binseg.configs.datasets.hrf1024 import dataset as hrf
from bob.ip.binseg.configs.datasets.chasedb11024 import dataset as chasedb
from bob.db.iostar import Database as IOSTAR
from bob.ip.binseg.data.transforms import *
import torch
from bob.ip.binseg.data.binsegdataset import BinSegDataset, SSLBinSegDataset, UnLabeledBinSegDataset


#### Config ####

# PyTorch dataset
labeled_dataset = torch.utils.data.ConcatDataset([drive,stare,hrf,chasedb])

#### Unlabeled IOSTAR Train ####
unlabeled_transforms = Compose([  
                        RandomHFlip()
                        ,RandomVFlip()
                        ,RandomRotation()
                        ,ColorJitter()
                        ,ToTensor()
                    ])

# bob.db.dataset init
iostarbobdb = IOSTAR(protocol='default_vessel')

# PyTorch dataset
unlabeled_dataset = UnLabeledBinSegDataset(iostarbobdb, split='train', transform=unlabeled_transforms)

# SSL Dataset

dataset = SSLBinSegDataset(labeled_dataset, unlabeled_dataset)