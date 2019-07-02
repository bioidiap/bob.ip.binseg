from bob.ip.binseg.configs.datasets.drive960 import dataset as drive
from bob.ip.binseg.configs.datasets.stare960 import dataset as stare
from bob.ip.binseg.configs.datasets.hrf960 import dataset as hrf
from bob.ip.binseg.configs.datasets.iostarvessel960 import dataset as iostar
import torch

#### Config ####

# PyTorch dataset
dataset = torch.utils.data.ConcatDataset([drive,stare,hrf,iostar])
