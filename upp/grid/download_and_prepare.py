# Downloads each container from the grid, and creates the VDS files for them
from __future__ import annotations

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from ftag.hdf5 import H5Reader
from ftag.vds import create_virtual_file

from upp.classes.preprocessing_config import PreprocessingConfig
from upp.grid.grid_split import get_output_ds_name


def check_container_exists(container: str) -> bool:
    command = ["rucio", "list-files", container]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False


def download_container(container: str, output_dir: Path | None):
    command = [
        "rucio",
        "download",
        container,
    ]
    print(f"Running command: {' '.join(command)}")
    try:
        subprocess.run(command, check=True, cwd=output_dir)
        return (container, True)
    except subprocess.CalledProcessError as e:
        print(f"Error downloading {container}: {e}")
        return (container, False)


def run_download(containers: list[str], output_dir: Path | None = None, max_parallel=4):
    try:
        subprocess.run(
            ["rucio", "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except FileNotFoundError:
        print("rucio command not found. Please setup with `lsetup rucio`")
        return False

    print(
        f"Downloading {len(containers)} containers to "
        f"{output_dir if output_dir else 'current directory'}"
    )
    try:
        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = [
                executor.submit(download_container, container, output_dir)
                for container in containers
            ]
            failed = []
            for future in as_completed(futures):
                container, success = future.result()
                if not success:
                    failed.append(container)
    except KeyboardInterrupt:
        print("\nDownload interrupted by user. Cancelling remaining tasks...")
        # Shutdown the executor immediately
        executor.shutdown(cancel_futures=True)
        return False
    except Exception as e:
        print(f"An error occurred during download: {e}")
        executor.shutdown(cancel_futures=True)
        return False

    if failed:
        print(f"Failed to download {len(failed)} containers:")
        for f in failed:
            print(f"- {f}")
        return False
    return True


def create_all_vds_files(
    containers: list[str],
    output_dir: Path | None = None,
):
    if output_dir is None:
        output_dir = Path.cwd()

    for i, container in enumerate(containers):
        cdir = output_dir / container
        vds_dir = cdir / "vds"
        vds_dir.mkdir(parents=True, exist_ok=True)
        create_virtual_file(
            str(cdir / "*.h5"),
            vds_dir / "vds.h5",
            overwrite=True,
        )
        print(
            f"[{i + 1}/{len(containers)}] Created VDS file for {container} at {vds_dir / 'vds.h5'}"
        )


def create_meta_data(
    config: Path,
    containers: list[str],
    output_dir: Path | None = None,
):
    if output_dir is None:
        output_dir = Path.cwd()

    pp_config = PreprocessingConfig.from_file(config, split="train", skip_checks=True)

    files_by_component: dict[str, dict] = {}

    for split in ["train", "val", "test"]:
        files_by_component[split] = {}
        for flavour in pp_config.components.flavours:
            sel_containers = [
                Path(output_dir) / c for c in containers if split in c and flavour.name in c
            ]

            vds = [f / "vds/vds.h5" for f in sel_containers]
            assert all(
                f.exists() for f in vds
            ), f"Not all VDS files exist for {split} {flavour}. Found: {vds}"

            files_by_component[split][flavour.name] = [str(f) for f in vds]

    output_file = output_dir / "organised-components.yaml"

    # Create a reader for each components to get num jets
    num_jets = {
        split: {
            flavour: H5Reader(
                files_by_component[split][flavour],
            ).num_jets
            for flavour in files_by_component[split]
        }
        for split in files_by_component
    }

    with open(output_file, "w") as f:
        yaml.dump(
            {
                "files": files_by_component,
                "num_jets": num_jets,
            },
            f,
            default_flow_style=False,
        )


def main():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--config",
        type=Path,
        help="Path to the preprocessing configuration file.",
        required=True,
    )
    args.add_argument(
        "--tag", type=str, required=False, help="The tag used when running the grid-submission"
    )
    args.add_argument(
        "--rucio_user",
        type=str,
        required=True,
        help="Rucio user to use for the submission",
    )
    args.add_argument(
        "--no-download",
        "-d",
        action="store_true",
        help="If set, will not download the containers from rucio. "
        "Assumes they are already downloaded.",
    )
    args.add_argument(
        "-n",
        "--n-threads",
        type=int,
        default=4,
        help="Number of parallel rucio download calls to run. Default is 4.",
    )
    args = args.parse_args()

    if not args.no_download:
        assert args.tag, "You must provide a tag if you want to download the containers."

    config = PreprocessingConfig.from_file(args.config, "train", skip_checks=True)
    containers_and_splits = PreprocessingConfig.get_input_files_with_split_components(args.config)
    split_containers = []
    for container, splits in containers_and_splits.items():
        output_dataset = get_output_ds_name(container, args.rucio_user, output_name=args.tag)
        print(f"Container: {container}, Output dataset: {output_dataset}")
        for split in splits:
            split_containers.append(f"{output_dataset}_output_{split}.h5")

    output_dir = Path(config.base_dir) / "split-components"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not split_containers:
        print("No containers found in the file.")
        return

    print(f"Looking for the following containers: {split_containers}")
    if not args.no_download:
        for i, container in enumerate(split_containers):
            if not check_container_exists(container):
                print(
                    f"Container {container} does not exist in Rucio. Please check the tag and user."
                )
                return
            else:
                print(f"[{i + 1}/{len(split_containers)}] Container {container} exists in Rucio.")

        if not run_download(split_containers, output_dir, args.n_threads):
            print("Failed to download one or more containers.")
            return

    print("Creating VDS files for the downloaded containers...")
    create_all_vds_files(
        split_containers,
        output_dir=output_dir,
    )

    create_meta_data(
        args.config,
        split_containers,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
