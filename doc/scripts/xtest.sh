#!/usr/bin/env bash

# Runs cross database tests

BOB=$HOME/work/bob/bob.ip.binseg/bin/bob

for d in drive stare chasedb1 iostar-vessel hrf; do
    for m in driu hed m2unet unet; do
        cmd=(${BOB} binseg analyze -vv ${m} "${d}-xtest")
        cmd+=("--weight=${m}/${d}/model/model_final.pth")
        cmd+=("--output-folder=${m}/${d}/xtest")
        "${cmd[@]}"
    done
done
