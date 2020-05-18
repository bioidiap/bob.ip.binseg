#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""MobileNetV2 U-Net model for image segmentation

The MobileNetV2 architecture is based on an inverted residual structure where
the input and output of the residual block are thin bottleneck layers opposite
to traditional residual models which use expanded representations in the input
an MobileNetV2 uses lightweight depthwise convolutions to filter features in
the intermediate expansion layer.  This model implements a MobileNetV2 U-Net
model, henceforth named M2U-Net, combining the strenghts of U-Net for medical
segmentation applications and the speed of MobileNetV2 networks.

References: [SANDLER-2018]_, [RONNEBERGER-2015]_
"""

from torch.optim.lr_scheduler import MultiStepLR
from bob.ip.binseg.models.m2unet import m2unet
from bob.ip.binseg.models.losses import SoftJaccardBCELogitsLoss
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

model = m2unet()

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
criterion = SoftJaccardBCELogitsLoss(alpha=0.7)

# scheduler
scheduler = MultiStepLR(
    optimizer, milestones=scheduler_milestones, gamma=scheduler_gamma
)
