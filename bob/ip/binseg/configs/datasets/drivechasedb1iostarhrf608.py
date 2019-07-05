from bob.ip.binseg.configs.datasets.drive608 import dataset as drive
from bob.ip.binseg.configs.datasets.chasedb1608 import dataset as chase
from bob.ip.binseg.configs.datasets.iostarvessel608 import dataset as iostar
from bob.ip.binseg.configs.datasets.hrf608 import dataset as hrf
import torch

#### Config ####

# PyTorch dataset
dataset = torch.utils.data.ConcatDataset([drive,chase,iostar,hrf])