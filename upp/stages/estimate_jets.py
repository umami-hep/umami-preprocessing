from __future__ import annotations

import logging as log
import math
import re
from pathlib import Path

from upp.classes.preprocessing_config import PreprocessingConfig
from upp.stages.hist import create_histograms
from upp.stages.resampling import Resampling
from upp.utils.logger import setup_logger


def run_estimate_jets(
    config_path: Path | str,
    output_path: Path | str,
    prep: bool = True,
) -> None:
    """Read the config and estimate max resampled jets.
    """
    setup_logger()

    output_path = Path(output_path)
    if output_path.parent:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = log.FileHandler(output_path, mode="w")
    try:

        class StripMarkupFormatter(log.Formatter):
            def format(self, record):
                message = super().format(record)
                return re.sub(r"\[/?[a-z_ ]+\]", "", message)

        file_handler.setFormatter(StripMarkupFormatter("%(message)s"))
        log.getLogger().addHandler(file_handler)

        config_path = Path(config_path)

        log.info(f"[bold green]Running estimate-jets on {config_path}...")

        # Load config while skipping checks and config copy
        config = PreprocessingConfig.from_file(
            config_path, "train", skip_checks=True, skip_config_copy=True, is_estimating_jets=True
        )

        if config.skip_resampling:
            log.warning("Resampling is skipped in this config. Cannot estimate resampled maximums.")
            return

        # Create 2D histograms (prep step)
        if prep:
            create_histograms(config)

        max_jets = {}

        for c in config.components:
            orig_num_jets = c.num_jets
            c.num_jets = 1_000_000_000_000

            resampling = Resampling(config)
            # Suppress stdout to avoid spam
            try:
                resampling.run(region=c.region.name, component=c.name)
            except ValueError as e:
                msg = str(e)
                match = re.search(r"but only ([\d,]+)", msg)
                if match:
                    num_str = match.group(1).replace(",", "")
                    max_jets[c.name] = int(num_str)
                else:
                    raise e

            c.num_jets = orig_num_jets
            log.info(f"Component {c.name} has a maximum of {max_jets[c.name]:,} resampled jets.")

        regions: dict = {}

        for c in config.components:
            if c.region.name not in regions:
                regions[c.region.name] = {}
            regions[c.region.name][c.flavour.name] = max_jets[c.name]

        if "lowpt" not in regions:
            lowpt_name = next(iter(regions.keys()))
            log.warning(
                f"No 'lowpt' region found. Using '{lowpt_name}' to find target flavour ratios."
            )
        else:
            lowpt_name = "lowpt"

        lowpt_counts = regions[lowpt_name]
        total_lowpt = sum(lowpt_counts.values())
        if total_lowpt == 0:
            raise ValueError(f"Total jets in region '{lowpt_name}' is 0. Aborting.")

        ratios = {f: c / total_lowpt for f, c in lowpt_counts.items()}
        log.info(f"[bold green]Flavour fractions in {lowpt_name} region:")
        for f, r in ratios.items():
            log.info(f"  {f}: {r:.4f}")

        new_counts_by_region_flavour = {}
        for r_name, r_counts in regions.items():
            if r_name == lowpt_name:
                for f, c in r_counts.items():
                    new_counts_by_region_flavour[(r_name, f)] = c
                continue

            max_total_for_r = min(r_counts[f] / ratios[f] for f in r_counts if ratios.get(f, 0) > 0)

            for f in r_counts:
                new_counts_by_region_flavour[(r_name, f)] = math.floor(max_total_for_r * ratios[f])

        log.info("[bold green]Calculated new maximum num_jets:")
        for (r, f), count in new_counts_by_region_flavour.items():
            log.info(f"  {r} {f}: {count:,}")
    finally:
        log.getLogger().removeHandler(file_handler)
        file_handler.close()
