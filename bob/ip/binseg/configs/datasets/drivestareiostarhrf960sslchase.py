from bob.ip.binseg.configs.datasets.drive960 import dataset as drive
from bob.ip.binseg.configs.datasets.stare960 import dataset as stare
from bob.ip.binseg.configs.datasets.hrf960 import dataset as hrf
from bob.ip.binseg.configs.datasets.iostarvessel960 import dataset as iostar
from bob.db.chasedb1 import Database as CHASE
from bob.db.hrf import Database as HRF
from bob.ip.binseg.data.transforms import *
import torch
from bob.ip.binseg.data.binsegdataset import BinSegDataset, SSLBinSegDataset, UnLabeledBinSegDataset


#### Config ####

# PyTorch dataset
labeled_dataset = torch.utils.data.ConcatDataset([drive,stare,hrf,iostar])

#### Unlabeled CHASE TRAIN ####
unlabeled_transforms = Compose([  
                        Crop(0,18,960,960)
                        ,RandomHFlip()
                        ,RandomVFlip()
                        ,RandomRotation()
                        ,ColorJitter()
                        ,ToTensor()
                    ])

# bob.db.dataset init
chasebobdb = CHASE(protocol = 'default')

# PyTorch dataset
unlabeled_dataset = UnLabeledBinSegDataset(chasebobdb, split='train', transform=unlabeled_transforms)

# SSL Dataset

dataset = SSLBinSegDataset(labeled_dataset, unlabeled_dataset)