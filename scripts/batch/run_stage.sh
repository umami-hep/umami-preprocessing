#!/usr/bin/env bash
# Runs one preprocessing stage inside the container. Called by slurm_batch.sh / condor_batch.sh.
set -e

usage() {
    cat <<EOF
Usage:
  run_stage.sh <config>
    # full chain (prep+resample+merge+norm+plot) with split=all

  run_stage.sh <config> normalise|normalize
  run_stage.sh <config> merge <split>
  run_stage.sh <config> plotting <split>
  run_stage.sh <config> prepare <component> <split>
  run_stage.sh <config> resampling <region> <split>
  run_stage.sh <config> fine_resampling <region> <component> <split>
EOF
}

if [ "$#" -eq 0 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
    exit 2
fi

CONFIG="$1"
shift

run_preprocess() {
    preprocess --config "${CONFIG}" "$@"
}

need_args() {
    # Usage: need_args <n_required> <mode> "$@"
    local n_required="$1"
    local mode="$2"
    shift 2
    if [ "$#" -lt "$n_required" ]; then
        echo "ERROR: '$mode' needs ${n_required} argument(s), got $#." >&2
        usage
        exit 2
    fi
}

if [ "$#" -eq 0 ]; then
    echo "No mode given. Processing full chain."
    run_preprocess --prep --split=all
    run_preprocess --resample --split=all
    run_preprocess --merge --split=all
    run_preprocess --norm
    run_preprocess --plot --split=all
    exit 0
fi

MODE="$1"
shift

case "${MODE}" in
    normalise|normalize)
        echo "Normalisation selected. Processing..."
        run_preprocess --norm
        ;;

    merge)
        need_args 1 "merge" "$@"
        echo "Start merging for $1. Processing..."
        run_preprocess --merge --split "$1"
        ;;

    plotting)
        need_args 1 "plotting" "$@"
        echo "Plotting selected ($1). Processing..."
        run_preprocess --plot --split "$1"
        ;;

    prepare)
        need_args 2 "prepare" "$@"
        echo "Start preparation for $1 ($2). Processing..."
        run_preprocess --prep --component "$1" --split "$2"
        ;;

    resampling)
        need_args 2 "resampling" "$@"
        echo "Start resampling for $1 ($2). Processing..."
        run_preprocess --resample --region "$1" --split "$2"
        ;;

    fine_resampling)
        need_args 3 "fine_resampling" "$@"
        echo "Start resampling for $2 ($3). Processing..."
        run_preprocess --resample --region "$1" --component "$2" --split "$3"
        ;;

    *)
        echo "Step '${MODE}' not supported!" >&2
        usage
        exit 2
        ;;
esac

echo "Done!"
