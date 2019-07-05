from bob.ip.binseg.configs.datasets.drive960all import dataset as drive
from bob.ip.binseg.configs.datasets.stare960all import dataset as stare
from bob.ip.binseg.configs.datasets.hrf960all import dataset as hrf
from bob.ip.binseg.configs.datasets.iostarvessel960all import dataset as iostar
from bob.db.chasedb1 import Database as CHASE
from bob.db.hrf import Database as HRF
from bob.ip.binseg.data.transforms import *
import torch


#### Config ####

# PyTorch dataset
dataset = torch.utils.data.ConcatDataset([drive,stare,hrf,iostar])