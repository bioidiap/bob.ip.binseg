# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Combining all optic disc dataset together with the same resolution."""

from ..drionsdb.expert1_768 import dataset as drionsdb
from ..drishtigs1.disc_all_768 import dataset as drishti
from ..iostar.optic_disc_768 import dataset as iostar
from ..refuge.disc_768 import dataset as refuge
from ..rimoner3.disc_exp1_768 import dataset as rimoner

dataset = drionsdb
keys = ["train", "test"]

dataset_list = [drishti, refuge, rimoner, iostar]

for database in dataset_list:
    for key in keys:
        dataset[key]._samples.extend(database[key]._samples)
