# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""HED Network for image segmentation.

Holistically-nested edge detection (HED), turns pixel-wise edge classification
into image-to-image prediction by means of a deep learning model that leverages
fully convolutional neural networks and deeply-supervised nets.

Reference: [XIE-2015]_
"""


from torch.optim.lr_scheduler import MultiStepLR

from deepdraw.engine.adabound import AdaBound
from deepdraw.models.hed import hed
from deepdraw.models.losses import MultiSoftJaccardBCELogitsLoss

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

model = hed()

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
criterion = MultiSoftJaccardBCELogitsLoss(alpha=0.7)

# scheduler
scheduler = MultiStepLR(
    optimizer, milestones=scheduler_milestones, gamma=scheduler_gamma
)
