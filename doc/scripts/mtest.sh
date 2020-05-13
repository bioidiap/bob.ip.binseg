#!/usr/bin/env bash

# Runs cross database tests

BOB=$HOME/work/bob/bob.ip.binseg/bin/bob

for d in drive stare chasedb1 iostar-vessel hrf; do
    for m in driu hed m2unet unet; do
        cmd=(${BOB} binseg analyze -vv ${m} "${d}-mtest")
        cmd+=("--weight=${m}/${d}/model/model_lowest_valid_loss.pth")
        cmd+=("--output-folder=${m}/${d}/mtest")
        "${cmd[@]}"
    done
done
