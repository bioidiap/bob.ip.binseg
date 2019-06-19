from bob.ip.binseg.configs.datasets.drive1168 import dataset as drive
from bob.ip.binseg.configs.datasets.stare1168 import dataset as stare
from bob.ip.binseg.configs.datasets.chasedb11168 import dataset as chase
from bob.ip.binseg.configs.datasets.iostarvessel1168 import dataset as iostar
import torch

#### Config ####

# PyTorch dataset
dataset = torch.utils.data.ConcatDataset([drive,stare,chase,iostar])