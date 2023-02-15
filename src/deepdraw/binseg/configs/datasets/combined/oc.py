# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Tim Laibacher, tim.laibacher@idiap.ch
# SPDX-FileContributor: Oscar Jiménez del Toro, oscar.jimenez@idiap.ch
# SPDX-FileContributor: Maxime Délitroz, maxime.delitroz@idiap.ch
# SPDX-FileContributor: Andre Anjos andre.anjos@idiap.ch
# SPDX-FileContributor: Daniel Carron, daniel.carron@idiap.ch
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Combining all optic cup dataset together with the same resolution."""
from ..drishtigs1.cup_all_768 import dataset as drishti
from ..refuge.cup_768 import dataset as refuge
from ..rimoner3.cup_exp1_768 import dataset as rimoner

dataset = drishti
keys = ["train", "test"]

dataset_list = [refuge, rimoner]

for database in dataset_list:
    for key in keys:
        dataset[key]._samples.extend(database[key]._samples)
