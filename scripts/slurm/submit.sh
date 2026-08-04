#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${UPP_IMAGE:-docker://gitlab-registry.cern.ch/aft/training-images/upp-images/upp:latest}"

# Throttle between sbatch calls (seconds). Set to 0 to disable.
THROTTLE="${THROTTLE:-30}"

# Dry-run: if 1, print commands but do not execute sbatch.
DRY_RUN="${DRY_RUN:-0}"

CONFIG=""

# Filters on the components enumerated from the config; empty means "everything".
declare -a REGION_FILTER=()
declare -a SAMPLE_FILTER=()
declare -a FLAV_FILTER=()
declare -a SPLIT_FILTER=()

# ---- Helpers --------------------------------------------------------------
slugify() {
  # keep alnum, dash, underscore; replace others with dash; squish repeats; trim
  local s="${*:-}"
  s="${s//[^[:alnum:]_-]/-}"
  s="$(printf '%s' "$s" | sed -E 's/-+/-/g; s/^-+//; s/-+$//')"
  # Slurm JobName limit is 128 chars; leave margin
  printf '%.*s' 120 "$s"
}

build_job_name() {
  local mode="${1:-sequential}"
  shift || true
  local prefix="upp-$(basename "${CONFIG%.*}")"
  case "$mode" in
    prepare)         slugify "${prefix}-prepare-${1:-component}-${2:-split}" ;;
    fine_resampling) slugify "${prefix}-fres-${1:-region}-${2:-component}-${3:-split}" ;;
    resampling)      slugify "${prefix}-resampling-${1:-region}-${2:-split}" ;;
    merge)           slugify "${prefix}-merge-${1:-split}" ;;
    normalise|normalize) slugify "${prefix}-normalise" ;;
    plotting)        slugify "${prefix}-plotting-${1:-split}" ;;
    sequential|*)    slugify "${prefix}-seq" ;;
  esac
}

enumerate() {
  # Use a local UPP installation when available, otherwise the container image
  if command -v list_components >/dev/null 2>&1; then
    list_components --config "${CONFIG}" "$@"
  else
    apptainer exec "${IMAGE}" list_components --config "${CONFIG}" "$@"
  fi
}

append_list() {
  # Append a comma/space separated list to the named array
  local input="$1"
  local out_name="$2"
  local -a items=()
  local x
  IFS=', ' read -r -a items <<< "$input"
  for x in "${items[@]}"; do
    if [[ -n "$x" ]]; then
      eval "$out_name+=(\"\$x\")"
    fi
  done
}

add_unique() {
  # Append the value to the named array if not already present
  local x="$1"
  local out_name="$2"
  local -a arr=()
  eval "arr=(\"\${${out_name}[@]}\")"
  local i
  for i in "${arr[@]}"; do
    if [[ "$i" == "$x" ]]; then
      return 0
    fi
  done
  eval "$out_name+=(\"\$x\")"
}

in_list() {
  # in_list <value> [items...]; an empty list matches everything
  local x="$1"
  shift
  if [[ $# -eq 0 ]]; then
    return 0
  fi
  local i
  for i in "$@"; do
    if [[ "$i" == "$x" ]]; then
      return 0
    fi
  done
  return 1
}

print_resolved_config() {
  printf '\nResolved configuration:\n'
  printf '  CONFIG     : %s\n' "$CONFIG"
  printf '  IMAGE      : %s\n' "$IMAGE"
  printf '  THROTTLE   : %s\n' "$THROTTLE"
  printf '  COMPONENTS : %s\n' "${#COMPONENTS[@]}"
  printf '  SPLITS     : %s\n' "${SPLITS[*]}"
  printf '\n'
}

prompt_with_default() {
  # Usage: prompt_with_default "Question" "default value"
  local prompt="$1"
  local default="$2"
  local reply

  read -r -p "${prompt} [${default}]: " reply
  if [[ -z "${reply}" ]]; then
    printf '%s\n' "${default}"
  else
    printf '%s\n' "${reply}"
  fi
}

interactive_mode() {
  echo
  echo "No mode provided. Entering interactive mode."
  echo
  echo "Available modes:"
  echo "  sequential prepare fine_resampling resampling merge normalise plotting"
  echo

  MODE="$(prompt_with_default "Select mode" "prepare")"

  local regions samples flavs splits
  regions="$(prompt_with_default "Regions"  "${ALL_REGIONS[*]}")"
  samples="$(prompt_with_default "Samples"  "${ALL_SAMPLES[*]}")"
  flavs="$(prompt_with_default "Flavours" "${ALL_FLAVS[*]}")"
  splits="$(prompt_with_default "Splits"   "${SPLITS[*]}")"

  append_list "${regions}" REGION_FILTER
  append_list "${samples}" SAMPLE_FILTER
  append_list "${flavs}"   FLAV_FILTER
  SPLIT_FILTER=()
  append_list "${splits}"  SPLIT_FILTER
}

submit() {
  # Usage: submit [mode args...]
  local jobname
  jobname="$(build_job_name "$@")"

  local -a cmd=(
    sbatch
    --job-name="$jobname"
    --output="${PWD}/logs/%j_%x.out"
    --error="${PWD}/logs/%j_%x.err"
    "${SCRIPT_DIR}/batch.sh"
    "$CONFIG"
    "$@"
  )

  echo "Submitting: $jobname"

  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY-RUN: '
    printf '%q ' "${cmd[@]}"
    printf '\n'
  else
    "${cmd[@]}"
    if [[ "$THROTTLE" != "0" ]]; then
      sleep "$THROTTLE"
    fi
  fi
}

usage() {
  cat <<'EOF'
Usage:
  submit.sh --config <yaml> [options] [mode]

Modes:
  (no mode)              # interactive mode: prompts for mode and filters
  sequential             # one job running the full chain
  prepare                # one job per component and split
  fine_resampling        # one job per component and split
  resampling             # one job per region and split
  merge                  # one job per split
  normalise|normalize    # one job
  plotting               # one job per split

Options:
  --config <yaml>        # preprocessing config (required)
  --dry-run              # do not call sbatch; print commands instead
  --regions "<list>"     # only submit components in these regions
  --samples "<list>"     # only submit components from these samples
  --flavs "<list>"       # only submit components with these flavours
  --splits "<list>"      # only submit these splits (default: train val test)
  --throttle N           # seconds between sbatch calls (default 30; 0 disables)

Env (still supported):
  UPP_IMAGE=<uri>        # container image (docker:// URI or local .sif)
  THROTTLE=<seconds>     # same as --throttle
  DRY_RUN=1              # same as --dry-run

List format:
  Comma and/or space separated, e.g. "lowpt,highpt" or "lowpt highpt"

Examples:
  ./submit.sh --config configs/my-config.yaml --dry-run prepare
  ./submit.sh --config configs/my-config.yaml --regions lowpt --splits train prepare
  ./submit.sh --config configs/my-config.yaml resampling
EOF
}

parse_args() {
  local -a rest=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help)
        usage
        exit 0
        ;;
      --config)
        [[ $# -ge 2 ]] || { echo "ERROR: --config requires a value" >&2; exit 2; }
        CONFIG="$2"
        shift 2
        ;;
      --dry-run)
        DRY_RUN=1
        shift
        ;;
      --throttle)
        [[ $# -ge 2 ]] || { echo "ERROR: --throttle requires a value" >&2; exit 2; }
        THROTTLE="$2"
        shift 2
        ;;
      --regions)
        [[ $# -ge 2 ]] || { echo "ERROR: --regions requires a value" >&2; exit 2; }
        append_list "$2" REGION_FILTER
        shift 2
        ;;
      --samples)
        [[ $# -ge 2 ]] || { echo "ERROR: --samples requires a value" >&2; exit 2; }
        append_list "$2" SAMPLE_FILTER
        shift 2
        ;;
      --flavs|--flavors|--flavours)
        [[ $# -ge 2 ]] || { echo "ERROR: --flavs requires a value" >&2; exit 2; }
        append_list "$2" FLAV_FILTER
        shift 2
        ;;
      --splits)
        [[ $# -ge 2 ]] || { echo "ERROR: --splits requires a value" >&2; exit 2; }
        append_list "$2" SPLIT_FILTER
        shift 2
        ;;
      --) # end of options
        shift
        rest+=("$@")
        break
        ;;
      -*)
        echo "ERROR: unknown option: $1" >&2
        usage
        exit 2
        ;;
      *)
        rest+=("$1")
        shift
        ;;
    esac
  done

  ARGS_REST=("${rest[@]}")
}

# ---- Main -----------------------------------------------------------------
main() {
  declare -a ARGS_REST=()
  parse_args "$@"
  set -- "${ARGS_REST[@]}"

  if [[ -z "$CONFIG" ]]; then
    echo "ERROR: --config is required" >&2
    usage
    exit 2
  fi

  # Enumerate all components defined in the config (TSV: region sample flavour name)
  local -a ALL_ROWS=()
  mapfile -t ALL_ROWS < <(enumerate)
  if [[ ${#ALL_ROWS[@]} -eq 0 ]]; then
    echo "ERROR: no components found in $CONFIG" >&2
    exit 1
  fi

  local -a ALL_REGIONS=() ALL_SAMPLES=() ALL_FLAVS=()
  local row region sample flavour name
  for row in "${ALL_ROWS[@]}"; do
    IFS=$'\t' read -r region sample flavour name <<< "$row"
    add_unique "$region" ALL_REGIONS
    add_unique "$sample" ALL_SAMPLES
    add_unique "$flavour" ALL_FLAVS
  done

  MODE="${1:-}"
  declare -a SPLITS=(train val test)
  if [[ -z "$MODE" ]]; then
    interactive_mode
  fi
  if [[ ${#SPLIT_FILTER[@]} -gt 0 ]]; then
    SPLITS=("${SPLIT_FILTER[@]}")
  fi

  # Apply the filters
  declare -a COMPONENTS=() REGIONS=()
  for row in "${ALL_ROWS[@]}"; do
    IFS=$'\t' read -r region sample flavour name <<< "$row"
    if in_list "$region" "${REGION_FILTER[@]}" \
        && in_list "$sample" "${SAMPLE_FILTER[@]}" \
        && in_list "$flavour" "${FLAV_FILTER[@]}"; then
      COMPONENTS+=("$row")
      add_unique "$region" REGIONS
    fi
  done

  if [[ "$DRY_RUN" == "1" ]]; then
    print_resolved_config
  fi

  if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p "${PWD}/logs"
  fi

  local split
  case "$MODE" in
    sequential)
      echo "Start submission for sequential processing..."
      submit
      ;;

    prepare)
      echo "Start submission for preparation..."
      for row in "${COMPONENTS[@]}"; do
        IFS=$'\t' read -r region sample flavour name <<< "$row"
        for split in "${SPLITS[@]}"; do
          submit "prepare" "$name" "$split"
        done
      done
      ;;

    fine_resampling)
      echo "Start submission for fine resampling..."
      for row in "${COMPONENTS[@]}"; do
        IFS=$'\t' read -r region sample flavour name <<< "$row"
        for split in "${SPLITS[@]}"; do
          submit "fine_resampling" "$region" "$name" "$split"
        done
      done
      ;;

    resampling)
      echo "Start submission for resampling..."
      for region in "${REGIONS[@]}"; do
        for split in "${SPLITS[@]}"; do
          submit "resampling" "$region" "$split"
        done
      done
      ;;

    merge)
      echo "Start submission for merging..."
      for split in "${SPLITS[@]}"; do
        submit "merge" "$split"
      done
      ;;

    normalise|normalize)
      echo "Start submission for normalise..."
      submit "normalise"
      ;;

    plotting)
      echo "Start submission for plotting..."
      for split in "${SPLITS[@]}"; do
        submit "plotting" "$split"
      done
      ;;

    *)
      echo "Unsupported mode: '$MODE'" >&2
      usage
      exit 2
      ;;
  esac
  echo "Done!"
}

main "$@"
