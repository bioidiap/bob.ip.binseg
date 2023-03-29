#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Tim Laibacher, tim.laibacher@idiap.ch
# SPDX-FileContributor: Oscar Jiménez del Toro, oscar.jimenez@idiap.ch
# SPDX-FileContributor: Maxime Délitroz, maxime.delitroz@idiap.ch
# SPDX-FileContributor: Andre Anjos andre.anjos@idiap.ch
# SPDX-FileContributor: Daniel Carron, daniel.carron@idiap.ch
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Runs cross database tests

BOB=$HOME/work/bob/deepdraw/bin/bob

for d in drive stare chasedb1 iostar-vessel hrf; do
    for m in driu hed m2unet unet; do
        cmd=(${BOB} binseg analyze -vv ${m} "${d}-mtest")
        cmd+=("--weight=${m}/${d}/model/model_lowest_valid_loss.pth")
        cmd+=("--output-folder=${m}/${d}/mtest")
        "${cmd[@]}"
    done
done
