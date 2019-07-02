from bob.ip.binseg.configs.datasets.stare544 import dataset as stare
from bob.ip.binseg.configs.datasets.chasedb1544 import dataset as chase
from bob.ip.binseg.configs.datasets.iostarvessel544 import dataset as iostar
from bob.ip.binseg.configs.datasets.hrf544 import dataset as hrf
import torch

#### Config ####

# PyTorch dataset
dataset = torch.utils.data.ConcatDataset([stare,chase,hrf,iostar])