#!/usr/bin/env python

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Tim Laibacher, tim.laibacher@idiap.ch
# SPDX-FileContributor: Oscar Jiménez del Toro, oscar.jimenez@idiap.ch
# SPDX-FileContributor: Maxime Délitroz, maxime.delitroz@idiap.ch
# SPDX-FileContributor: Andre Anjos andre.anjos@idiap.ch
# SPDX-FileContributor: Daniel Carron, daniel.carron@idiap.ch
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Residual U-Net for image segmentation.

A semantic segmentation neural network which combines the strengths of residual
learning and U-Net is proposed for road area extraction.  The network is built
with residual units and has similar architecture to that of U-Net. The benefits
of this model is two-fold: first, residual units ease training of deep
networks. Second, the rich skip connections within the network could facilitate
information propagation, allowing us to design networks with fewer parameters
however better performance.

Reference: [ZHANG-2017]_
"""

from bob.ip.binseg.engine.adabound import AdaBound
from bob.ip.binseg.models.losses import SoftJaccardBCELogitsLoss
from bob.ip.binseg.models.resunet import resunet50
from torch.optim.lr_scheduler import MultiStepLR

# config
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

model = resunet50()

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
