#!/usr/bin/env bash

# Runs all of our baselines

# set output directory and location of "bob" executable
OUTDIR=/idiap/temp/aanjos/binseg/baselines-with-validation
BOB=/idiap/user/aanjos/work/bob/bob.ip.binseg/bin/bob

# run <modelconfig> <dbconfig> <batchsize> [<device> [<queue>]]
function run() {
    local device="cpu"
    [ $# -gt 3 ] && device="${4}"

    local cmd=(${BOB} binseg experiment)
    cmd+=("-vv" "--device=${device}" ${1} ${2})
    cmd+=("--batch-size=${3}" "--output-folder=${OUTDIR}/${1}/${2}")

    [ $# -gt 4 ] && cmd=(jman submit "--name=$(basename ${OUTDIR})-${1}-${2}" "--memory=24G" "--queue=${5}" -- "${cmd[@]}")

    "${cmd[@]}"
}

# run/submit all baselines
# comment out from "sgpu/gpu" to run locally
# comment out from "cuda:0" to run on CPU
run m2unet drive 16 cuda:0 sgpu
run hed drive 8 cuda:0 sgpu
run driu drive 8 cuda:0 sgpu
run unet drive 4 cuda:0 sgpu
run m2unet stare 6 cuda:0 sgpu
run hed stare 4 cuda:0 sgpu
run driu stare 5 cuda:0 sgpu
run unet stare 2 cuda:0 sgpu
run m2unet chasedb1 6 cuda:0 sgpu
run hed chasedb1 4 cuda:0 sgpu
run driu chasedb1 4 cuda:0 sgpu
run unet chasedb1 2 cuda:0 sgpu
run m2unet hrf 1 cuda:0 gpu
run hed hrf 1 cuda:0 gpu
run driu hrf 1 cuda:0 gpu
run unet hrf 1 cuda:0 gpu
run m2unet iostar-vessel 6 cuda:0 sgpu
run hed iostar-vessel 4 cuda:0 sgpu
run driu iostar-vessel 4 cuda:0 sgpu
run unet iostar-vessel 2 cuda:0 sgpu
