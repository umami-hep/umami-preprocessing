#!/usr/bin/env bash
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem-per-cpu 8000
#SBATCH --time 1-00:00:00
# Edit the header above for your cluster, e.g.
# #SBATCH --partition <your-partition>
# #SBATCH --account <your-account>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${UPP_IMAGE:-docker://gitlab-registry.cern.ch/aft/training-images/upp-images/upp:latest}"
BINDS="${UPP_BINDS:-/home,/tmp}"

srun apptainer exec --contain --pwd "${PWD}" -B "${BINDS}" \
    "${IMAGE}" "${SCRIPT_DIR}/run_stage.sh" "$@"
