# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Faster R-CNN for Object Detection.

Reference: [____]_
"""
import torch

from torch.optim.lr_scheduler import MultiStepLR

from deepdraw.detect.models.faster_rcnn import faster_rcnn
from deepdraw.detect.models.losses import SoftJaccardBCELogitsLoss

# config
lr = 0.005
betas = (0.9, 0.999)
eps = 1e-08
weight_decay = 0.0005
final_lr = 0.1
gamma = 1e-3
eps = 1e-8
amsbound = False

scheduler_milestones = [900]
scheduler_gamma = 0.1

model = faster_rcnn()

params = [p for p in model.parameters() if p.requires_grad]

# optimizer
optimizer = torch.optim.SGD(
    params, lr=lr, momentum=0.9, weight_decay=weight_decay
)

# scheduler
scheduler = MultiStepLR(
    optimizer, milestones=scheduler_milestones, gamma=scheduler_gamma
)

# criterion
criterion = SoftJaccardBCELogitsLoss(alpha=0.7)
