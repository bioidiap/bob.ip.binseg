#!/usr/bin/env bash

# Runs all of our baselines

# set output directory and location of "bob" executable
OUTDIR=/path/where/to/dump/results
BOB=/path/to/bob

# run <modelconfig> <dbconfig> <batchsize> [<device> [<queue>]]
function run() {
    local device="cpu"
    [ $# -gt 3 ] && device="${4}"

    local cmd=(${BOB} binseg experiment)
    cmd+=("-vv" "--device=${device}" ${1} ${2})
    cmd+=("--batch-size=${3}" "--output-folder=${OUTDIR}/${1}/${2}")

    mkdir -pv ${OUTDIR}/${1}/${2}

    [ $# -gt 4 ] && cmd=(jman submit "--log-dir=${OUTDIR}/${1}/${2}" "--name=$(basename ${OUTDIR})-${1}-${2}" "--memory=24G" "--queue=${5}" -- "${cmd[@]}")

    if [ $# -le 4 ]; then
        # executing locally, capture stdout and stderr
        ("${cmd[@]}" | tee "${OUTDIR}/${1}/${2}/stdout.log") 3>&1 1>&2 2>&3 | tee "${OUTDIR}/${1}/${2}/stderr.log"
    else
        "${cmd[@]}"
    fi
}


# run/submit all baselines
# comment out from "cuda:0" to run on CPU
# comment out from "sgpu/gpu" to run locally
run m2unet drive         16 #cuda:0 #sgpu
run hed    drive          8 #cuda:0 #sgpu
run driu   drive          8 #cuda:0 #sgpu
run unet   drive          4 #cuda:0 #sgpu
run lwnet  drive          4 #cuda:0 #sgpu
run m2unet stare          6 #cuda:0 #sgpu
run hed    stare          4 #cuda:0 #sgpu
run driu   stare          5 #cuda:0 #sgpu
run unet   stare          2 #cuda:0 #sgpu
run lwnet  stare          4 #cuda:0 #sgpu
run m2unet chasedb1       6 #cuda:0 #sgpu
run hed    chasedb1       4 #cuda:0 #sgpu
run driu   chasedb1       4 #cuda:0 #sgpu
run unet   chasedb1       2 #cuda:0 #sgpu
run lwnet  chasedb1       4 #cuda:0 #sgpu
run m2unet hrf            1 #cuda:0 # gpu
run hed    hrf            1 #cuda:0 # gpu
run driu   hrf            1 #cuda:0 # gpu
run unet   hrf            1 #cuda:0 # gpu
run lwnet  hrf            4 #cuda:0 # gpu
run m2unet iostar-vessel  6 #cuda:0 # gpu
run hed    iostar-vessel  4 #cuda:0 # gpu
run driu   iostar-vessel  4 #cuda:0 # gpu
run unet   iostar-vessel  2 #cuda:0 # gpu
run lwnet  iostar-vessel  4 #cuda:0 # gpu
