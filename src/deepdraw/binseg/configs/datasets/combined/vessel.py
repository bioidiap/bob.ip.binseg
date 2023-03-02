# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Combining all vessel dataset together with the same resolution."""

from ..chasedb1.first_annotator_768 import dataset as chasedb
from ..drive.default_768 import dataset as drive
from ..hrf.default_768 import dataset as hrf
from ..iostar.vessel_768 import dataset as iostar
from ..stare.ah_768 import dataset as stare

dataset = drive
keys = ["train", "test"]

dataset_list = [stare, chasedb, hrf, iostar]

for database in dataset_list:
    for key in keys:
        dataset[key]._samples.extend(database[key]._samples)
