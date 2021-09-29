#!/usr/bin/env bash

# set output directory and location of "bob" executable
OUTDIR=/path/where/to/dump/results
BOB=/path/to/bob

# this function just makes running/submitting jobs a bit easier for extensive
# benchmark jobs.
# usage: run <modelconfig> <dbconfig> <batchsize> [<device> [<queue>]]
function run() {
    local device="cpu"
    [ $# -gt 3 ] && device="${4}"

    local cmd=(${BOB} binseg experiment)
    cmd+=("-vv" "--device=${device}" ${1} ${2})
    cmd+=("--batch-size=${3}" "--output-folder=${OUTDIR}/${1}/${2}")
    # add --multiproc-data-loading=0 to increase data loading/transform speeds,
    # but pay by making your results harder to reproduce (OS-random data loading)
    #cmd+=("--multiproc-data-loading=0")

    mkdir -pv ${OUTDIR}/${1}/${2}

    [ $# -gt 4 ] && cmd=(jman submit "--log-dir=${OUTDIR}/${1}/${2}" "--name=$(basename ${OUTDIR})-${1}-${2}" "--memory=24G" "--queue=${5}" -- "${cmd[@]}")

    if [ $# -le 4 ]; then
        # executing locally, capture stdout and stderr
        ("${cmd[@]}" | tee "${OUTDIR}/${1}/${2}/stdout.log") 3>&1 1>&2 2>&3 | tee "${OUTDIR}/${1}/${2}/stderr.log"
    else
        "${cmd[@]}"
    fi
}
