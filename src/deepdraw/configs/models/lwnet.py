# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Little W-Net for image segmentation.

The Little W-Net architecture contains roughly around 70k parameters and
closely matches (or outperforms) other more complex techniques.

Reference: [GALDRAN-2020]_
"""

from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from deepdraw.models.losses import MultiWeightedBCELogitsLoss
from deepdraw.models.lwnet import LittleWNet

# config
max_lr = 0.01  # start
min_lr = 1e-08  # valley
cycle = 5  # epochs for a complete scheduling cycle

model = LittleWNet()

criterion = MultiWeightedBCELogitsLoss()
model.criterion = criterion
model.valid_criterion = criterion

optimizer = Adam(
    model.parameters(),
    lr=max_lr,
)
model.optimizer = optimizer

scheduler = CosineAnnealingLR(
    optimizer,
    T_max=cycle,
    eta_min=min_lr,
)
model.scheduler = scheduler
