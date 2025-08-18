from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

import yaml

from upp.classes.preprocessing_config import PreprocessingConfig

BASE_DIR = Path(__file__).parent.parent.parent
print("Base dir:", BASE_DIR)


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, _data):
        return True


# Force inner lists to be written in flow style ([a, b, c])
def represent_list_as_flow(dumper, data):
    if all(isinstance(i, list) for i in data):
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=False)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


# Register representer
NoAliasDumper.add_representer(list, represent_list_as_flow)


def create_submission_dir(
    submision_dir: Path,
    config: Path,
):
    # Create a new config for each dataset with its cuts
    if submision_dir.exists():
        print("Submission directory already exists, removing it:", submision_dir)
        subprocess.run(["rm", "-rf", str(submision_dir)], check=True)

    print("Creating submission directory:", submision_dir)
    submision_dir.mkdir(parents=True, exist_ok=True)

    env_dir = BASE_DIR / "env"

    if not env_dir.exists():
        print("Environment directory does not exist, please run setup_env.sh first.")
        return False
    code_dir = BASE_DIR / "upp"
    # Copy the environment to the submission directory

    print("Creating symlink to environment in submission directory:", submision_dir / "myenv")
    (submision_dir / "env").symlink_to(env_dir, target_is_directory=True)
    (submision_dir / "upp").symlink_to(code_dir, target_is_directory=True)

    # Copy the configs directory to the submission directory
    subprocess.run(
        ["cp", str(config), "config.yaml"],
        cwd=submision_dir,
    )

    # Make the tarball
    print("Running ls")
    subprocess.run(["ls"])

    tarball_cmd = [
        "tar",
        "--exclude=*.pyc",
        "--exclude=*.pyo",
        "--dereference",
        "--hard-dereference",
        "-czf",
        "submission.tar.gz",
    ] + list(str(s.name) for s in submision_dir.glob("*"))

    print("Running command:", shlex.join(tarball_cmd))
    print()
    subprocess.run(tarball_cmd, cwd=submision_dir, check=True)
    return True


def get_output_ds_name(
    input_ds: str,
    rucio_user: str,
    output_name: str = "FTag_RW",
    output_ds: str = "user.{user}.{dsid}.{tags}.{name}_split.h5",
):
    split_input_ds = input_ds.split(".")

    if len(split_input_ds) < 4:
        print(
            "Expected input dataset format: user.{user}.{dsid}.{tags}.name "
            f"but got {input_ds}. Using default dsid and no tag."
        )
        dsid = "0000000"
        tags = "no_tag"
    else:
        try:
            dsid = str(int(input_ds.split(".")[2]))
        except ValueError:
            print(
                "Expected input dataset format: user.{user}.{dsid}.{tags}.name "
                f"but got {input_ds}. Using default dsid and no tag."
            )
            dsid = "0000000"

        tags = input_ds.split(".")[3]

    return output_ds.format(user=rucio_user, dsid=dsid, tags=tags, name=output_name)


def submit(
    config: Path,
    rucio_user: str,
    output_name: str = "FTag_RW",
    output_ds: str = "user.{user}.{dsid}.{tags}.{name}_split.h5",
    dryrun: bool = True,
    prun_args: list[str] | None = None,
):
    submision_dir = Path("submission")
    if not create_submission_dir(
        submision_dir=Path("submission"),
        config=config,
    ):
        return

    all_output_datasets = []

    # PP config will complain if this directory does not exist, but we don't care about it
    with open(config) as f:
        loaded_config = yaml.safe_load(f)
        ntuple_dir = Path(loaded_config["global"]["base_dir"]).resolve() / loaded_config[
            "global"
        ].get("ntuple_dir", "ntuples")
        Path(ntuple_dir).mkdir(parents=True, exist_ok=True)
        print("Using ntuple directory:", ntuple_dir)
        # Search all second level objects for a key 'pattern'
        all_patterns = []
        for _key, value in loaded_config.items():
            if isinstance(value, dict) and "pattern" in value:
                all_patterns.append(value["pattern"])
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and "pattern" in item:
                        if isinstance(item["pattern"], list):
                            all_patterns += item["pattern"]
                        else:
                            all_patterns.append(item["pattern"])
        # Flatten the list of patterns
        all_patterns = [pattern for sublist in all_patterns for pattern in sublist]
        for pattern in all_patterns:
            pat_dir = ntuple_dir / Path(pattern)
            # create an empty file not directory for each
            pat_dir.touch()

    containers_with_cuts = PreprocessingConfig.get_input_files_with_split_components(
        config.resolve()
    )

    for dataset, cuts_by_component in containers_with_cuts.items():
        print(f"Submitting dataset: {dataset}")
        # Create a config file for each dataset
        outputs = ",".join([f"output_{split}.h5" for split in cuts_by_component])
        final_output_ds = get_output_ds_name(dataset, rucio_user, output_name, output_ds)

        all_output_datasets.extend(
            [f"{final_output_ds}_output_{split}.h5" for split in cuts_by_component]
        )

        exec = " ".join(
            [
                "export PYTHONPATH=$PWD:$PYTHONPATH;",
                "./env/bin/python upp/main.py",
                "--no-resample",
                "--no-merge",
                "--no-norm",
                "--no-plot",
                "--split",
                "train",
                "--split-components",
                "--config",
                "config.yaml",
                "--files %IN",
                "--container",
                dataset,
                "--grid",
            ]
        )

        # Prepare the command to submit the job
        command = [
            "prun",
            "--inTarBall",
            str(submision_dir / "submission.tar.gz"),
            "--inDS",
            dataset,
            "--outDS",
            get_output_ds_name(dataset, rucio_user, output_name, output_ds),
            "--exec",
            exec,
            "--mergeScript",
            "hdf5-merge-nolock -o %OUT -i %IN",
            "--forceStaged",
            "--forceStagedSecondary",
            "--useAthenaPackages",
            "--unlimitNumOutputs",
            "--outputs",
            outputs,
            "--maxNFilesPerJob",
            "5",
            "--nGBPerJob",
            "20",
            "--nGBPerMergeJob",
            "20",
            "--respectSplitRule",
        ] + (prun_args or [])
        print("Running command:", shlex.join(command))
        if not dryrun:
            subprocess.run(command, check=True)

    with open(submision_dir / "output_datasets.txt", "w") as f:
        for output_ds in all_output_datasets:
            f.write(f"{output_ds}\n")


def main():
    parser = argparse.ArgumentParser(description="Run the split files on grid")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/afs/cern.ch/user/n/npond/try_pp_on_grid/ftag-rw/configs/single-b.yaml"),
        help="Path to the config file",
    )
    parser.add_argument(
        "--rucio_user",
        type=str,
        required=True,
        help="Rucio user to use for the submission",
    )
    parser.add_argument(
        "--tag",
        type=str,
        required=True,
        help="Tag for the file names",
    )
    parser.add_argument(
        "--dryrun",
        "-d",
        action="store_true",
        default=False,
        help="If set, the script will not submit the job but only create the submission directory",
    )

    args, prun_args = parser.parse_known_args()
    print("Additional arguments for submission:", prun_args)

    # Check if the prun command is available
    try:
        subprocess.run(
            ["prun", "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError:
        print("prun command not found. Please setup with `lsetup panda`")
        return
    submit(
        config=args.config,
        rucio_user=args.rucio_user,
        output_name=args.tag,
        dryrun=args.dryrun,
        prun_args=prun_args,
    )


if __name__ == "__main__":
    main()
