#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""DRIU Network for Vessel Segmentation using SSL and Batch Normalization

Deep Retinal Image Understanding (DRIU), a unified framework of retinal image
analysis that provides both retinal vessel and optic disc segmentation using
deep Convolutional Neural Networks (CNNs).  This version of our model includes
a loss that is suitable for Semi-Supervised Learning (SSL).  This version also
includes batch normalization as a regularization mechanism.

Reference: [MANINIS-2016]_
"""

from torch.optim.lr_scheduler import MultiStepLR
from bob.ip.binseg.models.driu_bn import driu_bn
from bob.ip.binseg.models.losses import MixJacLoss
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

model = driu_bn()

# optimizer
optimizer = AdaBound(
    model.parameters(),
    lr=lr,
    betas=betas,
    final_lr=final_lr,
    gamma=gamma,
    eps=eps,
    weight_decay=weight_decay,
    amsbound=amsbound,
)

# criterion
criterion = MixJacLoss(lambda_u=0.05, jacalpha=0.7)
ssl = True

# scheduler
scheduler = MultiStepLR(
    optimizer, milestones=scheduler_milestones, gamma=scheduler_gamma
)
