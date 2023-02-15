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

"""Little W-Net for image segmentation.

The Little W-Net architecture contains roughly around 70k parameters and
closely matches (or outperforms) other more complex techniques.

Reference: [GALDRAN-2020]_
"""

from bob.ip.binseg.models.losses import MultiWeightedBCELogitsLoss
from bob.ip.binseg.models.lwnet import lwnet
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# config
max_lr = 0.01  # start
min_lr = 1e-08  # valley
cycle = 50  # epochs for a complete scheduling cycle

model = lwnet()

criterion = MultiWeightedBCELogitsLoss()

optimizer = Adam(
    model.parameters(),
    lr=max_lr,
)

scheduler = CosineAnnealingLR(
    optimizer,
    T_max=cycle,
    eta_min=min_lr,
)
