#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch.optim.lr_scheduler import MultiStepLR
from bob.ip.binseg.modeling.driu import build_driu
import torch.optim as optim
from torch.nn import BCEWithLogitsLoss
from bob.ip.binseg.utils.model_zoo import modelurls

##### Config #####
pretrained_weight = 'vgg16'

lr = 0.001
betas = (0.9, 0.999)
eps = 1e-08
weight_decay = 0
amsgrad = False

scheduler_milestones = [150]
scheduler_gamma = 0.1

# model
model = build_driu()

# pretrained backbone
pretrained_backbone = modelurls['vgg16']

# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
    
# criterion
criterion = BCEWithLogitsLoss()

# scheduler
scheduler = MultiStepLR(optimizer, milestones=scheduler_milestones, gamma=scheduler_gamma)
