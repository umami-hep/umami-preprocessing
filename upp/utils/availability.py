"""Estimate the available objects per component and recommend object counts.

The estimate is the expensive part (one pass over the input samples per component),
so it is cached. The recommendation itself is pure arithmetic and is redone on every
call, which means the sample weights or the target class ratios can be changed without
measuring again.
"""

from __future__ import annotations

import argparse
import hashlib
import logging as log
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import yaml
from ftag.cli_utils import HelpFormatter, valid_path

from upp import __version__
from upp.classes.preprocessing_config import PreprocessingConfig
from upp.utils.logger import banner, setup_logger

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Generator

    from ftag import Cuts

    from upp.classes.components import Component, Components
    from upp.classes.preprocessing_config import Split

SPLITS = ("train", "val", "test")

# Name of the cache file written next to the preprocessing output
CACHE_FNAME = "availability.yaml"

# Tolerance factor applied in Component.get_auto_sampling_fraction(). Requesting more
# than available/TOLERANCE objects makes the automatic sampling fraction exceed one.
SAMPLING_TOLERANCE = 1.1


def cache_path(config: PreprocessingConfig) -> Path:
    """Return the path of the availability cache for a given config.

    Parameters
    ----------
    config : PreprocessingConfig
        Loaded preprocessing config

    Returns
    -------
    Path
        Path of the availability cache file
    """
    return config.out_dir / CACHE_FNAME


def cache_key(component: Component, cuts: Cuts, num_estimate: int | None) -> str:
    """Return a key identifying the inputs an estimate was made from.

    Parameters
    ----------
    component : Component
        Component the estimate belongs to
    cuts : Cuts
        Cuts that were applied for the estimate
    num_estimate : int | None
        Number of objects used for the estimate, None if all of them were used

    Returns
    -------
    str
        Key identifying the inputs of the estimate
    """
    pattern = component.sample.pattern
    patterns = [pattern] if isinstance(pattern, str) else list(pattern)
    parts = [
        *sorted(str(entry) for entry in patterns),
        *sorted(str(cut) for cut in cuts),
        f"equal_global_objects={component.equal_global_objects}",
        f"num_estimate={num_estimate}",
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def selected_fractions(
    stream: Generator, name: str, split_cuts: dict[str, Cuts]
) -> dict[str, float]:
    """Return the fraction of the streamed objects passing the cuts of each split.

    The objects are counted batch by batch, so only one batch is held in memory at a
    time. Summing the counts gives the same fraction as applying the cuts to all of the
    streamed objects at once.

    Parameters
    ----------
    stream : Generator
        Stream of batches to count
    name : str
        Name of the global object dataset in the batches
    split_cuts : dict[str, Cuts]
        Cuts of each split

    Returns
    -------
    dict[str, float]
        Fraction of the streamed objects passing the cuts, for each split
    """
    selected = dict.fromkeys(split_cuts, 0)
    total = 0
    for batch in stream:
        objects = batch[name]
        total += len(objects)
        for split, cuts in split_cuts.items():
            selected[split] += len(cuts(objects).values)
    return {split: num / total for split, num in selected.items()}


def estimate_availability(
    component: Component,
    split_cuts: dict[str, Cuts],
    num_estimate: int | None,
) -> dict[str, int]:
    """Estimate the available objects of one component for several splits.

    Mirrors ``H5Reader.estimate_available_global_objects()``, but evaluates the cuts of
    every split on the same streamed objects, so that all splits cost a single pass and
    no more than one batch is held in memory.

    Parameters
    ----------
    component : Component
        Component with an already set up reader
    split_cuts : dict[str, Cuts]
        Cuts of this component for each split
    num_estimate : int | None
        Number of objects to base the estimate on, None to use all of them

    Returns
    -------
    dict[str, int]
        Estimated number of available objects for each split
    """
    reader = component.reader
    name = reader.global_objects_name
    variables = list(dict.fromkeys(v for cuts in split_cuts.values() for v in cuts.variables))

    # reset rngs to ensure same objects are used for each sample
    reader.rng = np.random.default_rng(42)
    for single_reader in reader.readers:
        single_reader.rng = np.random.default_rng(42)

    # if equal_global_objects is True, available objects is based on the smallest sample
    if reader.equal_global_objects:
        totals: dict[str, list[float]] = {split: [] for split in split_cuts}
        for single_reader in reader.readers:
            fractions = selected_fractions(
                single_reader.stream({name: variables}, num_estimate), name, split_cuts
            )
            for split, fraction in fractions.items():
                totals[split].append(fraction * single_reader.num_global_objects)
        return {
            split: math.floor(min(total) * len(reader.readers) * 0.99)
            for split, total in totals.items()
        }

    # otherwise, available objects is based on all samples
    fractions = selected_fractions(reader.stream({name: variables}, num_estimate), name, split_cuts)
    return {
        split: math.floor(fraction * reader.num_global_objects * 0.99)
        for split, fraction in fractions.items()
    }


def sampling_factor(config: PreprocessingConfig, component: Component) -> float:
    """Return the fraction of the available objects the resampling can actually use.

    The resampling only selects ``sampling_fraction`` of the objects it reads, so
    requesting more than that fraction fails the check in ``check_num_global_objects``.
    With an automatic sampling fraction the limit is instead set by the tolerance factor
    which would otherwise push the fraction above one.

    Parameters
    ----------
    config : PreprocessingConfig
        Loaded preprocessing config
    component : Component
        Component to return the factor for

    Returns
    -------
    float
        Fraction of the available objects that is usable
    """
    if config.skip_resampling or component.is_target(config.sampl_cfg.target):
        return 1.0

    sampling_fraction = config.sampl_cfg.sampling_fraction
    if sampling_fraction in ("auto", None):
        return 1 / SAMPLING_TOLERANCE
    return min(float(sampling_fraction), 1.0)


def solve_counts(
    available: dict[str, dict[tuple[str, str], int]],
    weights: dict[str, int],
    ratios: dict[str, float] | None = None,
) -> dict[str, dict[tuple[str, str], int]]:
    """Solve for the object counts which use the available objects best.

    The counts of a region are the counts of a single weight unit scaled by the weight of
    that region, which keeps the class ratios identical across regions (as required by
    ``Components.check_flavour_ratios``) and the region ratios exactly at the requested
    weights.

    Parameters
    ----------
    available : dict[str, dict[tuple[str, str], int]]
        Available objects per split, keyed by (region, class)
    weights : dict[str, int]
        Relative weight of each region
    ratios : dict[str, float] | None, optional
        Target ratios of the classes. By default None, which maximises each class
        independently so that the ratios follow the available objects.

    Returns
    -------
    dict[str, dict[tuple[str, str], int]]
        Recommended object counts per split, keyed by (region, class)
    """
    counts: dict[str, dict[tuple[str, str], int]] = {}
    for split, split_available in available.items():
        classes = list(dict.fromkeys(name for _, name in split_available))

        # objects per weight unit, limited by the region with the fewest available objects
        units = {
            name: min(
                split_available[(region, name)] / weights[region]
                for region in weights
                if (region, name) in split_available
            )
            for name in classes
        }

        if ratios is None:
            per_unit = {name: int(unit) for name, unit in units.items()}
        else:
            scale = min(units[name] / ratios[name] for name in classes)
            per_unit = {name: int(scale * ratios[name]) for name in classes}

        counts[split] = {
            (region, name): per_unit[name] * weights[region] for region, name in split_available
        }

    return counts


def region_weights(config: PreprocessingConfig) -> dict[str, int]:
    """Return the relative weight of each region, taken from its sample config.

    Parameters
    ----------
    config : PreprocessingConfig
        Loaded preprocessing config

    Returns
    -------
    dict[str, int]
        Weight of each region

    Raises
    ------
    ValueError
        If the samples of one region define different weights
    """
    weights: dict[str, int] = {}
    for component in config.config["components"]:
        region = component["region"]["name"]
        weight = component["sample"].get("sample_weight", 1)
        if weights.setdefault(region, weight) != weight:
            raise ValueError(
                f"Region {region} is used with different sample_weight values "
                f"({weights[region]} and {weight}) - the weight has to be the same "
                "for all samples of a region."
            )
    return weights


def target_ratios(config: PreprocessingConfig) -> dict[str, float] | None:
    """Return the requested class ratios, or None if they are not fixed.

    Parameters
    ----------
    config : PreprocessingConfig
        Loaded preprocessing config

    Returns
    -------
    dict[str, float] | None
        Requested class ratios, None if each class is to be maximised independently
    """
    ratios = config.config.get("auto_counts", {}).get("class_ratios", "auto")
    return None if ratios == "auto" else dict(ratios)


def read_cache(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Read the cached estimates from file.

    Parameters
    ----------
    path : Path
        Path of the cache file

    Returns
    -------
    dict[str, dict[str, dict[str, Any]]]
        Cached estimates per component and split, empty if there is no cache yet
    """
    if not path.exists():
        return {}
    with open(path) as file:
        cache = yaml.safe_load(file)
    if (version := cache.get("upp_version")) != __version__:
        log.warning(
            f"The estimate in {path} was written with UPP {version}, this is UPP "
            f"{__version__}. Rerun estimate_object_counts if the numbers look wrong."
        )
    return cache.get("components", {})


def write_cache(path: Path, config: PreprocessingConfig, components: dict) -> None:
    """Write the estimates to the cache file.

    Parameters
    ----------
    path : Path
        Path of the cache file
    config : PreprocessingConfig
        Loaded preprocessing config the estimates belong to
    components : dict
        Estimates per component and split
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    cache = {
        "config": str(config.config_path),
        "upp_version": __version__,
        "num_global_objects_estimate_available": config.num_global_objects_estimate_available,
        "components": components,
    }
    with open(path, "w") as file:
        yaml.dump(cache, file, sort_keys=False)
    log.info(f"Wrote availability cache to {path}")


def get_availability(
    config_path: Path,
    splits: tuple[str, ...] = SPLITS,
    force: bool = False,
) -> tuple[dict[str, PreprocessingConfig], dict[str, dict[tuple[str, str], int]]]:
    """Return the available objects per split, measuring the ones which are not cached.

    Parameters
    ----------
    config_path : Path
        Path to the preprocessing config
    splits : tuple[str, ...], optional
        Splits to return the available objects for, by default all of them
    force : bool, optional
        Measure again even if a valid cache entry exists, by default False

    Returns
    -------
    tuple[dict[str, PreprocessingConfig], dict[str, dict[tuple[str, str], int]]]
        Config of each split and the available objects per split, keyed by (region, class)
    """
    configs = {
        split: PreprocessingConfig.from_file(
            config_path, cast("Split", split), skip_config_copy=True, skip_auto_counts=True
        )
        for split in splits
    }
    config = configs[splits[0]]
    num_estimate = config.num_global_objects_estimate_available
    if num_estimate is not None and num_estimate <= 0:
        num_estimate = None

    cached = {} if force else read_cache(cache_path(config))
    cache: dict[str, dict[str, dict[str, Any]]] = {}
    available: dict[str, dict[tuple[str, str], int]] = {split: {} for split in splits}

    for component in config.components:
        split_cuts = {
            split: next(c for c in configs[split].components if c.name == component.name).cuts
            for split in splits
        }
        keys = {
            split: cache_key(component, cuts, num_estimate) for split, cuts in split_cuts.items()
        }
        entry = cached.get(component.name, {})
        missing = {
            split: cuts
            for split, cuts in split_cuts.items()
            if entry.get(split, {}).get("key") != keys[split]
        }

        if missing:
            log.info(f"Estimating available objects for {component}...")
            component.setup_reader(
                config.batch_size,
                global_name=config.global_name,
                transform=config.transform,
            )
            estimates = estimate_availability(component, missing, num_estimate)
        else:
            log.info(f"Using cached estimate for {component}")
            estimates = {}

        cache[component.name] = {}
        for split in splits:
            num = cast("int", estimates.get(split, entry.get(split, {}).get("available")))
            cache[component.name][split] = {"available": num, "key": keys[split]}
            available[split][(component.region.name, component.flavour.name)] = num

    write_cache(cache_path(config), config, cache)
    return configs, available


def apply_sampling_factor(
    config: PreprocessingConfig,
    components: Components,
    available: dict[tuple[str, str], int],
) -> dict[tuple[str, str], int]:
    """Scale the available objects of one split by the fraction the resampling can use.

    Parameters
    ----------
    config : PreprocessingConfig
        Loaded preprocessing config of the split
    components : Components
        Components of the split
    available : dict[tuple[str, str], int]
        Available objects, keyed by (region, class)

    Returns
    -------
    dict[tuple[str, str], int]
        Usable objects, keyed by (region, class)
    """
    factors = {(c.region.name, c.flavour.name): sampling_factor(config, c) for c in components}
    return {key: int(num * factors[key]) for key, num in available.items()}


def usable_objects(
    configs: dict[str, PreprocessingConfig],
    available: dict[str, dict[tuple[str, str], int]],
) -> dict[str, dict[tuple[str, str], int]]:
    """Scale the available objects by the fraction the resampling can use.

    Parameters
    ----------
    configs : dict[str, PreprocessingConfig]
        Loaded preprocessing config of each split
    available : dict[str, dict[tuple[str, str], int]]
        Available objects per split, keyed by (region, class)

    Returns
    -------
    dict[str, dict[tuple[str, str], int]]
        Usable objects per split, keyed by (region, class)
    """
    return {
        split: apply_sampling_factor(configs[split], configs[split].components, split_available)
        for split, split_available in available.items()
    }


def resolve_auto_counts(config: PreprocessingConfig, components: Components) -> None:
    """Set the object counts of components which requested them automatically.

    Uses the estimate written by ``estimate_object_counts``, so no objects are read here.
    The solved counts are recorded in the config which is copied to the output directory.

    Parameters
    ----------
    config : PreprocessingConfig
        Loaded preprocessing config
    components : Components
        Components to set the object counts of

    Raises
    ------
    ValueError
        If the estimate of a component is missing or was made with different cuts
    """
    split = config.split
    cached = read_cache(cache_path(config))
    num_estimate = config.num_global_objects_estimate_available
    if num_estimate is not None and num_estimate <= 0:
        num_estimate = None

    available = {}
    for component in components:
        entry = cached.get(component.name, {}).get(split, {})
        if entry.get("key") != cache_key(component, component.cuts, num_estimate):
            raise ValueError(
                f"No up to date estimate of the available objects of {component} for the"
                f" {split} split. Run 'estimate_object_counts --config"
                f" {config.config_path}' to create or update {cache_path(config)}."
            )
        available[(component.region.name, component.flavour.name)] = entry["available"]

    counts = solve_counts(
        {split: apply_sampling_factor(config, components, available)},
        region_weights(config),
        target_ratios(config),
    )[split]

    resolved = {}
    for component in components:
        component.num_global_objects = counts[(component.region.name, component.flavour.name)]
        resolved[component.name] = component.num_global_objects
        log.info(f"Using {component.num_global_objects:,} objects for {component}")

    config.config.setdefault("auto_counts", {})[f"solved_{split}"] = resolved


def recommend(
    config_path: Path,
    splits: tuple[str, ...] = SPLITS,
    force: bool = False,
) -> tuple[dict[str, dict[tuple[str, str], int]], dict[str, dict[tuple[str, str], int]]]:
    """Estimate the available objects and recommend the object counts to request.

    Parameters
    ----------
    config_path : Path
        Path to the preprocessing config
    splits : tuple[str, ...], optional
        Splits to recommend counts for, by default all of them
    force : bool, optional
        Measure again even if a valid cache entry exists, by default False

    Returns
    -------
    tuple[dict[str, dict[tuple[str, str], int]], dict[str, dict[tuple[str, str], int]]]
        Available objects and recommended counts per split, keyed by (region, class)
    """
    configs, available = get_availability(config_path, splits, force)
    config = configs[splits[0]]
    counts = solve_counts(
        usable_objects(configs, available),
        region_weights(config),
        target_ratios(config),
    )
    return available, counts


def log_recommendation(
    config_path: Path,
    available: dict[str, dict[tuple[str, str], int]],
    counts: dict[str, dict[tuple[str, str], int]],
) -> None:
    """Log the available objects, the recommended counts and a config snippet.

    Parameters
    ----------
    config_path : Path
        Path to the preprocessing config
    available : dict[str, dict[tuple[str, str], int]]
        Available objects per split, keyed by (region, class)
    counts : dict[str, dict[tuple[str, str], int]]
        Recommended counts per split, keyed by (region, class)
    """
    log.info(banner(" Available objects "))
    for split, split_available in available.items():
        for (region, name), num in split_available.items():
            requested = counts[split][(region, name)]
            log.info(
                f"{split:>5} {region:>10} / {name:<14} {requested:>15,} requested"
                f" of {num:>15,} available"
            )

    keys = {
        "train": "num_global_objects",
        "val": "num_global_objects_val",
        "test": "num_global_objects_test",
    }
    log.info(banner(" Config snippet "))
    log.info(f"Add these to the components of {config_path.name}:")
    for region, name in counts[next(iter(counts))]:
        lines = [f"# {region} / {name}"]
        lines += [
            f"{keys[split]}: {counts[split][(region, name)]:_}" for split in counts if split in keys
        ]
        log.info("\n".join(lines))


def parse_args(args: Any) -> argparse.Namespace:
    """Parse the command line arguments.

    Parameters
    ----------
    args : Any
        Command line arguments.

    Returns
    -------
    argparse.Namespace
        Namespace with the parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=HelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        type=valid_path,
        help="Path to config file",
    )
    parser.add_argument(
        "--splits",
        default=",".join(SPLITS),
        type=lambda value: tuple(value.split(",")),
        help="Comma-separated list of splits to estimate the available objects for",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Estimate again instead of using the cached numbers",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        type=str.upper,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level. Overrides the UPP_LOG_LEVEL environment variable.",
    )

    return parser.parse_args(args)


def main(args: Any | None = None) -> None:
    """Estimate the available objects and log the recommended object counts.

    Parameters
    ----------
    args : Any | None, optional
        Command line arguments, by default None
    """
    args = parse_args(args)
    setup_logger(args.log_level)

    available, counts = recommend(args.config, args.splits, args.force)
    log_recommendation(args.config, available, counts)


if __name__ == "__main__":
    main()
