from bob.ip.binseg.configs.datasets.stare544 import dataset as stare
from bob.ip.binseg.configs.datasets.chasedb1544 import dataset as chase
from bob.ip.binseg.configs.datasets.iostarvessel544 import dataset as iostar
from bob.ip.binseg.configs.datasets.hrf544 import dataset as hrf
from bob.db.drive import Database as DRIVE
from bob.ip.binseg.data.transforms import *
import torch
from bob.ip.binseg.data.binsegdataset import BinSegDataset, SSLBinSegDataset, UnLabeledBinSegDataset


#### Config ####

# PyTorch dataset
labeled_dataset = torch.utils.data.ConcatDataset([stare,chase,iostar,hrf])

#### Unlabeled STARE TRAIN ####
unlabeled_transforms = Compose([  
                        CenterCrop((544,544))
                        ,RandomHFlip()
                        ,RandomVFlip()
                        ,RandomRotation()
                        ,ColorJitter()
                        ,ToTensor()
                    ])

# bob.db.dataset init
drivebobdb = DRIVE(protocol = 'default')

# PyTorch dataset
unlabeled_dataset = UnLabeledBinSegDataset(drivebobdb, split='train', transform=unlabeled_transforms)

# SSL Dataset

dataset = SSLBinSegDataset(labeled_dataset, unlabeled_dataset)