#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""U-Net for image segmentation

U-Net is a convolutional neural network that was developed for biomedical image
segmentation at the Computer Science Department of the University of Freiburg,
Germany.  The network is based on the fully convolutional network (FCN) and its
architecture was modified and extended to work with fewer training images and
to yield more precise segmentations.

Reference: [RONNEBERGER-2015]_
"""

from torch.optim.lr_scheduler import MultiStepLR
from bob.ip.binseg.models.unet import unet
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

model = unet()

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
