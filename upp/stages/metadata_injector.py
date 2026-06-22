from __future__ import annotations

import glob
import math
import shutil
from pathlib import Path

import h5py
from ftag.find_metadata import MetadataFinder
from numpy.lib import recfunctions as rfn

from upp.utils.logger import setup_logger


class MetadataInjector:
    def __init__(self, config):
        self.config = config
        self.log = setup_logger()

    def run(self):
        # Initialize input file list
        input_files = []
        if hasattr(self.config, "config") and "inputs" in self.config.config:
            raw_inputs = self.config.config["inputs"]
            if "train" in raw_inputs and "input_files" in raw_inputs["train"]:
                input_files.extend(raw_inputs["train"]["input_files"])

        # Expand wildcards/glob patterns
        expanded_files = []
        for f in input_files:
            matched = glob.glob(f)
            expanded_files.extend(matched)

        failures: list[tuple[str, str]] = []
        for fpath_str in expanded_files:
            fpath = Path(fpath_str)
            backup_path = fpath.with_suffix(fpath.suffix + ".bak")

            try:
                # 0. Create a physical backup
                shutil.copy(fpath, backup_path)

                # 1. Inject metadata (this adds a new 'metadata' group)
                finder = MetadataFinder(fpath_str)
                finder.inject_metadata()

                with h5py.File(fpath_str, "a") as f:
                    # 2. Calculate parameters required for physical weights
                    dsid = next(iter(f["metadata"].keys()))
                    xs = float(f[f"metadata/{dsid}/cross_section_pb"][()])
                    eff = float(f[f"metadata/{dsid}/genFiltEff"][()])
                    kf = float(f[f"metadata/{dsid}/kfactor"][()])

                    # Read Sum of Weights (SOW) with support for different formats
                    sow_ds = f["cutBookkeeper/nominal/counts"]
                    sow = (
                        float(sow_ds["sumOfWeights"][0]) if sow_ds.dtype.names else float(sow_ds[0])
                    )

                    # Guard against div-by-zero / non-finite metadata (silent inf/nan weights).
                    if sow == 0 or not math.isfinite(sow):
                        raise ValueError(f"Invalid sum-of-weights (sow={sow}) in {fpath_str}")
                    if not all(math.isfinite(v) for v in (xs, eff, kf)):
                        raise ValueError(
                            f"Non-finite metadata in {fpath_str}: xs={xs}, eff={eff}, kfactor={kf}"
                        )

                    # 3. Read original jets and their attributes
                    old_jets_ds = f["jets"]
                    original_attrs = dict(old_jets_ds.attrs)  # Backup all attribute metadata
                    jets_data = old_jets_ds[:]

                    # 4. Calculate Physical Weight
                    if "mcEventWeight" not in jets_data.dtype.names:
                        raise KeyError(f"mcEventWeight missing in {fpath_str}")

                    mcw = jets_data["mcEventWeight"].astype("f8")
                    # Formula: Weight = (XS * Efficiency * k-factor / SOW) * MC_Weight
                    physical_w = (xs * eff * kf / sow) * mcw

                    # 5. Construct the updated structured array
                    if "physicalWeight" in jets_data.dtype.names:
                        jets_data = rfn.drop_fields(jets_data, "physicalWeight")

                    updated_jets = rfn.append_fields(
                        jets_data, "physicalWeight", physical_w.astype("f4"), usemask=False
                    )

                    # 6. Delete and recreate the dataset while restoring attributes
                    del f["jets"]
                    new_ds = f.create_dataset("jets", data=updated_jets, compression="gzip")

                    # Restore all original attributes (e.g., descriptions for eventNumber, etc.)
                    for k, v in original_attrs.items():
                        new_ds.attrs[k] = v

                # 7. Processing successful: remove the backup
                backup_path.unlink()
                self.log.info(
                    f"Successfully updated {fpath.name} (Attributes and original fields preserved)"
                )

            except Exception as e:
                self.log.error(f"Failed for {fpath_str}: {e}")
                if backup_path.exists():
                    self.log.warning(f"Restoring {fpath.name} from backup...")
                    shutil.move(backup_path, fpath)
                failures.append((fpath_str, str(e)))

        # Re-raise a summary so partial failures are not silently missed in a long run.
        if failures:
            summary = "\n".join(f"  - {path}: {err}" for path, err in failures)
            raise RuntimeError(
                f"Metadata injection failed for {len(failures)}/{len(expanded_files)} file(s):\n"
                f"{summary}"
            )
