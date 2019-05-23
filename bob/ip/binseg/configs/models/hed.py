#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch.optim.lr_scheduler import MultiStepLR
from bob.ip.binseg.modeling.hed import build_hed
import torch.optim as optim
from bob.ip.binseg.modeling.losses import HEDSoftJaccardBCELogitsLoss
from bob.ip.binseg.utils.model_zoo import modelurls
from bob.ip.binseg.engine.adabound import AdaBound


##### Config #####
lr = 0.001
betas = (0.9, 0.999)
eps = 1e-08
weight_decay = 0
final_lr = 0.1
gamma = 1e-3
eps = 1e-8
amsbound = False

scheduler_milestones = [900]
scheduler_gamma = 0.1


# model
model = build_hed()

# pretrained backbone
pretrained_backbone = modelurls['vgg16']

# optimizer
optimizer = AdaBound(model.parameters(), lr=lr, betas=betas, final_lr=final_lr, gamma=gamma,
                 eps=eps, weight_decay=weight_decay, amsbound=amsbound) 
# criterion
criterion = HEDSoftJaccardBCELogitsLoss(alpha=0.7)

# scheduler
scheduler = MultiStepLR(optimizer, milestones=scheduler_milestones, gamma=scheduler_gamma)
