# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Semi-supervised pretrain dataset for the DRIVE dataset."""

from . import _semi_supervised_data_augmentation

dataset = _semi_supervised_data_augmentation("default", 768)
dataset["__extra_valid__"] = [dataset["test"]]