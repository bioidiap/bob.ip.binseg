from bob.ip.binseg.configs.datasets.drive1168 import dataset as drive
from bob.ip.binseg.configs.datasets.stare1168 import dataset as stare
from bob.ip.binseg.configs.datasets.chasedb11168 import dataset as chasedb
from bob.ip.binseg.configs.datasets.iostarvessel1168 import dataset as iostar
from bob.db.hrf import Database as HRF
from bob.ip.binseg.data.transforms import *
import torch
from bob.ip.binseg.data.binsegdataset import BinSegDataset, SSLBinSegDataset, UnLabeledBinSegDataset


#### Config ####

# PyTorch dataset
labeled_dataset = torch.utils.data.ConcatDataset([drive,stare,iostar,chasedb])

#### Unlabeled HRF TRAIN ####
unlabeled_transforms = Compose([  
                        RandomRotation()
                        ,Crop(0,108,2336,3296)
                        ,Resize((1168))
                        ,RandomHFlip()
                        ,RandomVFlip()
                        ,ColorJitter()
                        ,ToTensor()
                    ])

# bob.db.dataset init
hrfbobdb = HRF(protocol='default')

# PyTorch dataset
unlabeled_dataset = UnLabeledBinSegDataset(hrfbobdb, split='train', transform=unlabeled_transforms)

# SSL Dataset

dataset = SSLBinSegDataset(labeled_dataset, unlabeled_dataset)