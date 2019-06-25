from bob.ip.binseg.configs.datasets.drive1024 import dataset as drive
from bob.ip.binseg.configs.datasets.stare1024 import dataset as stare
from bob.ip.binseg.configs.datasets.hrf1024 import dataset as hrf
from bob.ip.binseg.configs.datasets.chasedb11024 import dataset as chase
import torch

#### Config ####

# PyTorch dataset
dataset = torch.utils.data.ConcatDataset([drive,stare,hrf,chase])
