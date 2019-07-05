from bob.ip.binseg.configs.datasets.drive608 import dataset as drive
from bob.ip.binseg.configs.datasets.chasedb1608 import dataset as chase
from bob.ip.binseg.configs.datasets.iostarvessel608 import dataset as iostar
from bob.ip.binseg.configs.datasets.hrf608 import dataset as hrf
from bob.db.stare import Database as STARE
from bob.ip.binseg.data.transforms import *
import torch
from bob.ip.binseg.data.binsegdataset import BinSegDataset, SSLBinSegDataset, UnLabeledBinSegDataset


#### Config ####

# PyTorch dataset
labeled_dataset = torch.utils.data.ConcatDataset([drive,chase,iostar,hrf])

#### Unlabeled STARE TRAIN ####
unlabeled_transforms = Compose([  
                        Pad((2,1,2,2))
                        ,RandomHFlip()
                        ,RandomVFlip()
                        ,RandomRotation()
                        ,ColorJitter()
                        ,ToTensor()
                    ])

# bob.db.dataset init
starebobdb = STARE(protocol = 'default')

# PyTorch dataset
unlabeled_dataset = UnLabeledBinSegDataset(starebobdb, split='train', transform=unlabeled_transforms)

# SSL Dataset

dataset = SSLBinSegDataset(labeled_dataset, unlabeled_dataset)