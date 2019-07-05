from bob.ip.binseg.configs.datasets.drive1024all import dataset as drive
from bob.ip.binseg.configs.datasets.stare1024all import dataset as stare
from bob.ip.binseg.configs.datasets.hrf1024all import dataset as hrf
from bob.ip.binseg.configs.datasets.chasedb11024all import dataset as chasedb
from bob.db.iostar import Database as IOSTAR
from bob.ip.binseg.data.transforms import *
import torch


#### Config ####

# PyTorch dataset
dataset = torch.utils.data.ConcatDataset([drive,stare,hrf,chasedb])

