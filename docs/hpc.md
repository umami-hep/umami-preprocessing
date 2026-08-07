# Running on HPC clusters

The preprocessing stages can be parallelized over components, regions and splits using the
`--component`, `--region` and `--split` flags described in [Run](run.md). On a Slurm or HTCondor
cluster, each of these units of work can run as its own batch job inside the
[container image](setup.md#container-image). UPP ships a small set of submission scripts in
`scripts/batch/` that automate this.

## Prerequisites

- A cluster with Slurm or HTCondor and apptainer. On lxplus, HTCondor is provided by the CERN
  batch service — see the [quickstart](https://batchdocs.web.cern.ch/local/quick.html) and
  [job submission](https://batchdocs.web.cern.ch/local/submit.html) documentation.
- A shared filesystem between the submitting node and the workers — the input, intermediate and
  output files must be visible to all jobs. This is the case on lxplus (AFS/EOS) and typical
  institute clusters.

!!!info "Input/output data on lxplus (AFS/EOS)"

    HTCondor on lxplus rejects submit files that reference EOS paths (executable, `output`,
    `error`, `log`), so create the run directory — with the copied scripts and the `logs/`
    directory — in your AFS work area and submit from there. The jobs themselves run with your
    Kerberos credentials and can read and write `/afs` and `/eos` directly, so keep the large
    input ntuples and outputs on EOS and bind both filesystems into the container:

    ```bash
    export UPP_BINDS=/afs,/eos,/tmp
    ```

    For very I/O-heavy workflows the batch service recommends staging data through the local
    pool space of the job instead of writing to EOS directly — see
    [Data flows](https://batchdocs.web.cern.ch/concepts/dataflow.html) and
    [EOS](https://batchdocs.web.cern.ch/troubleshooting/eos.html) in the CERN batch docs.

    If the run directory has to live on EOS, the experimental
    [EosSubmit schedds](https://batchdocs.web.cern.ch/local/eossubmit.html)
    (`module load lxbatch/eossubmit`) accept submit files with EOS paths, transferring all job
    files via xrootd instead of using a shared filesystem. All submit file paths must then be on
    EOS, and this mode has not been tested with these scripts.
- The UPP container image (see [Container image](setup.md#container-image)). The scripts default to
  the CVMFS-unpacked image
  `/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/aft/training-images/upp-images/upp:latest` when
  it exists (no download or conversion needed) and fall back to
  `docker://gitlab-registry.cern.ch/aft/training-images/upp-images/upp:latest` otherwise. Running
  from `docker://` directly is fine: apptainer caches the converted image and only downloads again
  when a new version is published. Alternatively, pin a specific local file with
  `export UPP_IMAGE=/path/to/upp.sif` after an `apptainer pull`.

!!!info "Apptainer cache location"

    The apptainer cache defaults to `~/.apptainer/cache` and the conversion uses `/tmp` as
    scratch space. On clusters with a small home quota (e.g. lxplus) point them to a larger
    filesystem:

    ```bash
    export APPTAINER_CACHEDIR=/path/to/big/storage/apptainer_cache
    export APPTAINER_TMPDIR=/path/to/big/storage/apptainer_tmp
    ```

## Interactive use

For quick tests, run UPP inside the container on an interactive allocation:

```bash
salloc --ntasks 1 --cpus-per-task 4 --time 2:00:00
srun apptainer exec --contain --pwd "$PWD" -B "$PWD" -B /home -B /tmp \
    "$UPP_IMAGE" preprocess --config <path/to/config.yaml> --prep
```

## Batch submission scripts

The `scripts/batch/` directory contains:

- `submit.sh` runs on the login node. It reads the components from your preprocessing config and
  submits one batch job per unit of work. The scheduler is auto-detected (`sbatch` found → Slurm,
  `condor_submit` found → HTCondor) and can be forced with `--scheduler slurm|condor`.
- `slurm_batch.sh` is the sbatch payload. It carries the `#SBATCH` resource header and starts the
  container on the compute node.
- `condor_job.sub` and `condor_batch.sh` are the HTCondor equivalents: the submit description with
  the resource requests, and the job executable starting the container.
- `run_stage.sh` runs inside the container and maps the submitted mode onto the `preprocess`
  command line flags.

To use them, create a run directory and copy the scripts. A clone of the repository is not
required — the image contains the repository at `/workspace`, so the scripts can be taken straight
from there:

=== "CVMFS"

    ```bash
    mkdir my_preprocessing && cd my_preprocessing
    cp -r /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/aft/training-images/upp-images/upp:latest/workspace/scripts/batch .
    ```

=== "apptainer"

    ```bash
    mkdir my_preprocessing && cd my_preprocessing
    apptainer exec docker://gitlab-registry.cern.ch/aft/training-images/upp-images/upp:latest \
        cp -r /workspace/scripts/batch .
    ```

=== "local clone"

    ```bash
    mkdir my_preprocessing && cd my_preprocessing
    cp -r <path/to/umami-preprocessing>/scripts/batch .
    ```

Adapt the resource specifications (number of CPUs, memory allocation, time limit, partition or
account etc.) in the `slurm_batch.sh` (Slurm) or `condor_job.sub` (HTCondor) files to fit your
needs. The `+JobFlavour` walltime flavours available on lxplus are listed in the
[CERN batch documentation](https://batchdocs.web.cern.ch/local/submit.html).

Then submit the stages in order, waiting for all jobs of one stage to finish before submitting the
next:

```bash
./batch/submit.sh --config <path/to/config.yaml> --dry-run prepare  # preview only
./batch/submit.sh --config <path/to/config.yaml> prepare
./batch/submit.sh --config <path/to/config.yaml> resampling
./batch/submit.sh --config <path/to/config.yaml> merge
./batch/submit.sh --config <path/to/config.yaml> normalise
./batch/submit.sh --config <path/to/config.yaml> plotting
```

Job logs are written to the `logs/output/` and `logs/error/` subdirectories of the current
directory. Running `submit.sh` without a mode enters
an interactive prompt for the mode and filters, and `./batch/submit.sh --help` prints all modes and
options.

On Slurm each job is submitted with its own `sbatch` call and job name. On HTCondor all jobs of one
`submit.sh` invocation are submitted as a single cluster (one `condor_submit` with one process per
job), with the job arguments written to `logs/condor_<mode>.args`.

The available modes and the jobs they submit:

| Mode | Jobs | `preprocess` flags per job |
|------|------|----------------------------|
| `sequential` | 1 | full chain (`--prep`, `--resample`, `--merge`, `--norm`, `--plot`) |
| `prepare` | one per component and split | `--prep --component <c> --split <s>` |
| `resampling` | one per region and split | `--resample --region <r> --split <s>` |
| `fine_resampling` | one per component and split | `--resample --region <r> --component <c> --split <s>` |
| `merge` | one per split | `--merge --split <s>` |
| `normalise` | 1 | `--norm` |
| `plotting` | one per split | `--plot --split <s>` |

!!!warning "Stage ordering and parallel h5py access"

    All jobs of a stage must finish before the next stage is submitted, e.g. all `prepare` jobs
    before `resampling`. Also run the [initial sample check](run.md#additional-scripts-initial-sample-check)
    once before submitting `prepare` jobs in parallel — it creates the virtual datasets which can
    get corrupted when created by multiple jobs at once.

## Config-driven job lists

`submit.sh` never hardcodes which components exist. It calls the `list_components` script (part of
UPP) to enumerate the components defined in the `components:` block of your config. No local UPP
installation is needed for this: when `list_components` is not on the `PATH`, it is run inside the
container image automatically. Only the `prepare`, `fine_resampling` and `resampling` modes (and
the interactive mode) enumerate at all — the other modes submit without running UPP on the login
node.

```bash
list_components --config <path/to/config.yaml>
```

```text
lowpt   ttbar   bjets   lowpt_ttbar_bjets
highpt  zprime  bjets   highpt_zprime_bjets
...
```

Only combinations actually defined in the config are submitted. The selection can be narrowed with
filter flags, each taking a comma- or space-separated list:

```bash
./batch/submit.sh --config <path/to/config.yaml> --regions lowpt --splits train prepare
./batch/submit.sh --config <path/to/config.yaml> --samples ttbar --flavs "bjets,cjets" fine_resampling
```

Note that enumerating the components fully validates the config, so a broken config fails directly
on the login node instead of inside the batch jobs.

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `UPP_IMAGE` | CVMFS-unpacked image if present, else `docker://...upp-images/upp:latest` | Container image (unpacked directory, local `.sif` or `docker://` URI) |
| `UPP_BINDS` | `/home,/tmp` | Comma-separated paths bound into the container (the run and script directories are always bound in addition) |
| `THROTTLE` | `30` | Seconds between `sbatch` calls (`0` disables; Slurm only) |
| `DRY_RUN` | `0` | Set to `1` to print the submission commands instead of submitting |

Make sure `UPP_BINDS` covers your input ntuples and output directory if they live outside `/home`
(e.g. on a scratch filesystem), and export `UPP_IMAGE`/`UPP_BINDS` in your shell so they are also
picked up by the batch jobs.

!!!warning "Keep the throttle enabled"

    The delay between `sbatch` calls avoids hammering the scheduler and gives jobs time to start
    up without all of them hitting the shared filesystem at once. Only disable it for small
    submissions.

!!!info "Configs outside the repository"

    When you copy a config out of the repository, `!include` directives with relative paths no
    longer resolve. Use absolute paths in `!include` lines of copied configs.
