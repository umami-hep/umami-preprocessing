#!/usr/bin/env bash
# HTCondor job executable: starts the container on the worker node.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Prefer the CVMFS-unpacked image when available, fall back to the registry
DEFAULT_IMAGE="/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/aft/training-images/upp-images/upp:latest"
if [[ ! -e "${DEFAULT_IMAGE}" ]]; then
    DEFAULT_IMAGE="docker://gitlab-registry.cern.ch/aft/training-images/upp-images/upp:latest"
fi
IMAGE="${UPP_IMAGE:-${DEFAULT_IMAGE}}"
BINDS="${UPP_BINDS:-/home,/tmp}"

apptainer exec --contain --pwd "${PWD}" -B "${BINDS}" -B "${PWD}" -B "${SCRIPT_DIR}" \
    "${IMAGE}" "${SCRIPT_DIR}/run_stage.sh" "$@"
