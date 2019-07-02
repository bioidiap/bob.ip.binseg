from bob.ip.binseg.configs.datasets.drive1168all import dataset as drive
from bob.ip.binseg.configs.datasets.stare1168all import dataset as stare
from bob.ip.binseg.configs.datasets.chasedb11168all import dataset as chasedb
from bob.ip.binseg.configs.datasets.iostarvessel1168all import dataset as iostar
from bob.db.hrf import Database as HRF
from bob.ip.binseg.data.transforms import *
import torch


#### Config ####

# PyTorch dataset
dataset = torch.utils.data.ConcatDataset([drive,stare,iostar,chasedb])