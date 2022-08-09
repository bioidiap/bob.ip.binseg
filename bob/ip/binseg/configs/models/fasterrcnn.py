#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Faster R-CNN for image detection

Reference: [____]_
"""
from torch.optim.lr_scheduler import MultiStepLR
from bob.ip.binseg.models.fasterrcnn import fasterrcnn
import torch

# config
lr = 0.005
betas = (0.9, 0.999)
eps = 1e-08
weight_decay = 0
final_lr = 0.1
gamma = 1e-3
eps = 1e-8
amsbound = False

scheduler_milestones = [900]
scheduler_gamma = 0.1

model = fasterrcnn()

# optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=lr,
                            momentum=0.9, weight_decay=0.0005)

# scheduler
scheduler = MultiStepLR(
    optimizer, milestones=scheduler_milestones, gamma=scheduler_gamma
)
