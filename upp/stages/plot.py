from __future__ import annotations

import logging as log
import re
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ftag import Cuts
from ftag.hdf5 import H5Reader
from puma import Histogram, HistogramPlot

from upp.utils.tools import path_append

if TYPE_CHECKING:  # pragma: no cover
    from upp.classes.plotting_config import PlottingConfig
    from upp.classes.preprocessing_config import PreprocessingConfig


@dataclass(frozen=True)
class PlotRegion:
    """Store one named plotting region.

    Attributes
    ----------
    name : str
        Name used in output file suffixes and log messages.
    cuts : Cuts
        Jet-level selection defining the region.
    pt_range : tuple[float, float] | None
        Raw pT range defining the region. Values follow the input ntuple units,
        usually MeV. If ``None``, the region has no explicit pT range.
    """

    name: str
    cuts: Cuts
    pt_range: tuple[float, float] | None = None


def _is_pt_variable(variable: str) -> bool:
    """Return whether a variable name represents transverse momentum.

    Parameters
    ----------
    variable : str
        Variable name from the preprocessing configuration.

    Returns
    -------
    bool
        ``True`` if the variable name contains ``pt`` case-insensitively.
    """
    return "pt" in variable.lower()


def _uses_gev_display_units(variable: str) -> bool:
    """Return whether a variable should be converted from MeV to GeV."""
    variable_lower = variable.lower()
    return "pt" in variable_lower or "mass" in variable_lower


def _display_values(variable: str, values: Any) -> Any:
    """Convert values to display units for plotting.

    Parameters
    ----------
    variable : str
        Variable being plotted.
    values : Any
        Values loaded from the HDF5 input.

    Returns
    -------
    Any
        Values converted to GeV for pT and mass variables and unchanged otherwise.
    """
    return values / 1e3 if _uses_gev_display_units(variable) else values


def _display_range(
    variable: str, bins_range: tuple[float, float] | None
) -> tuple[float, float] | None:
    """Convert a raw variable range to plotting units.

    Parameters
    ----------
    variable : str
        Variable associated with the range.
    bins_range : tuple[float, float] | None
        Raw range from config bins or component cuts.

    Returns
    -------
    tuple[float, float] | None
        Range converted to GeV for pT and mass variables and unchanged otherwise.
    """
    if bins_range is None:
        return None
    if _uses_gev_display_units(variable):
        return (bins_range[0] / 1e3, bins_range[1] / 1e3)
    return bins_range


def _range_from_bins(variable: str, bins: list[list[float]]) -> tuple[float, float]:
    """Return the full display range covered by configured bins.

    Parameters
    ----------
    variable : str
        Resampling variable name.
    bins : list[list[float]]
        Piecewise bin definition from the resampling config.

    Returns
    -------
    tuple[float, float]
        Display range from the first lower edge to the final upper edge.
    """
    display_range = _display_range(variable, (bins[0][0], bins[-1][1]))
    assert display_range is not None
    return display_range


def _suffix(*parts: str) -> str:
    """Build a filesystem-friendly suffix from name parts.

    Parameters
    ----------
    *parts : str
        Text parts to combine. Empty parts are ignored.

    Returns
    -------
    str
        Suffix beginning with ``_`` and containing only alphanumeric and
        underscore separators.
    """
    return "_" + "_".join(re.sub(r"[^A-Za-z0-9]+", "_", part).strip("_") for part in parts if part)


def _sample_label(sample_name: str, plotting: PlottingConfig) -> str:
    """Return a display label for a configured sample name.

    Parameters
    ----------
    sample_name : str
        Name of the configured input sample.
    plotting : PlottingConfig
        Active plotting configuration.

    Returns
    -------
    str
        LaTeX label for known samples and the raw sample name otherwise.
    """
    return plotting.sample_label(sample_name)


def _format_num_jets(num_jets: int) -> str:
    """Format a jet count with compact suffixes.

    Parameters
    ----------
    num_jets : int
        Number of jets to format.

    Returns
    -------
    str
        Compact count using ``k`` for thousands and ``M`` for millions.
    """
    if num_jets >= 1_000_000:
        value = num_jets / 1_000_000
        return f"{value:g}M"
    if num_jets >= 1_000:
        value = num_jets / 1_000
        return f"{value:g}k"
    return str(num_jets)


def _atlas_second_tag(
    *sample_names: str,
    plotting: PlottingConfig,
    num_jets: int | None = None,
    resampling_status: str | None = None,
) -> str:
    """Build the second ATLAS tag with energy, sample labels, status, and jet count.

    Parameters
    ----------
    *sample_names : str
        Sample names to include after the centre-of-mass energy.
    plotting : PlottingConfig
        Active plotting configuration.
    num_jets : int | None, optional
        Number of jets requested for plotting. If provided, it is added as an
        extra line using compact formatting.
    resampling_status : str | None, optional
        Resampling status added as an extra line.

    Returns
    -------
    str
        Multiline ATLAS second tag. Sample labels share the first line with the
        centre-of-mass energy, followed by the optional resampling status and
        jet count.
    """
    labels = [_sample_label(name, plotting) for name in dict.fromkeys(sample_names) if name]
    first_line = plotting.atlas_second_tag
    if labels:
        first_line = f"{first_line}, {' + '.join(labels)} jets"
    lines = [first_line]
    if resampling_status is not None:
        lines.append(resampling_status)
    if num_jets is not None:
        lines.append(f"{_format_num_jets(num_jets)} jets")
    return "\n".join(lines)


def _plotting_num_jets(config: PreprocessingConfig, available_jets: int) -> int:
    """Return the number of jets requested for plotting.

    Parameters
    ----------
    config : PreprocessingConfig
        Active preprocessing configuration.
    available_jets : int
        Number of jets available for the plotted selection.

    Returns
    -------
    int
        Minimum of the available jets and ``plotting.num_jets_plotting``.
    """
    assert config.plotting.num_jets_plotting is not None
    return min(available_jets, config.plotting.num_jets_plotting)


def _pt_bounds_from_cuts(cuts: Cuts, pt_variable: str) -> tuple[float, float] | None:
    """Extract lower and upper pT bounds from a cut collection.

    Parameters
    ----------
    cuts : Cuts
        Jet-level cuts associated with a component region.
    pt_variable : str
        Name of the pT variable used for resampling.

    Returns
    -------
    tuple[float, float] | None
        Raw lower and upper pT bounds if both are present; otherwise ``None``.
    """
    lower = None
    upper = None
    for cut in cuts:
        if cut.variable != pt_variable:
            continue
        if cut.operator in {">", ">="}:
            lower = cut.value if lower is None else max(lower, cut.value)
        elif cut.operator in {"<", "<="}:
            upper = cut.value if upper is None else min(upper, cut.value)
    if lower is None or upper is None:
        return None
    return (lower, upper)


def _cuts_from_pt_range(pt_variable: str, pt_range: tuple[float, float]) -> Cuts:
    """Create cuts selecting a raw pT interval.

    Parameters
    ----------
    pt_variable : str
        Name of the pT variable used for the selection.
    pt_range : tuple[float, float]
        Raw lower and upper pT bounds.

    Returns
    -------
    Cuts
        Selection requiring ``lower < pt_variable < upper``.
    """
    return Cuts.from_list(
        [
            [pt_variable, ">", pt_range[0]],
            [pt_variable, "<", pt_range[1]],
        ]
    )


def _pt_variable(config: PreprocessingConfig) -> str | None:
    """Find the configured resampling pT variable.

    Parameters
    ----------
    config : PreprocessingConfig
        Active preprocessing configuration.

    Returns
    -------
    str | None
        First resampling variable whose name looks like pT, or ``None`` if no
        pT variable is configured.
    """
    for variable in config.sampl_cfg.vars:
        if _is_pt_variable(variable):
            return variable
    return None


def _variable_range(config: PreprocessingConfig, variable: str) -> tuple[float, float]:
    """Return the display range for a configured resampling variable.

    Parameters
    ----------
    config : PreprocessingConfig
        Active preprocessing configuration.
    variable : str
        Resampling variable name.

    Returns
    -------
    tuple[float, float]
        Display range derived from the variable's configured bins.
    """
    return _range_from_bins(variable, config.sampl_cfg.bins[variable])


def _unique_regions(config: PreprocessingConfig, pt_variable: str | None) -> list[PlotRegion]:
    """Collect unique component regions for plotting.

    Parameters
    ----------
    config : PreprocessingConfig
        Active preprocessing configuration.
    pt_variable : str | None
        Configured pT variable, if available.

    Returns
    -------
    list[PlotRegion]
        Region selections with any explicit pT ranges extracted from their cuts.
    """
    regions = []
    seen = set()
    for region, _components in config.components.groupby_region():
        pt_range = _pt_bounds_from_cuts(region.cuts, pt_variable) if pt_variable else None
        key = (region.name, pt_range)
        if key in seen:
            continue
        seen.add(key)
        regions.append(PlotRegion(region.name, region.cuts, pt_range))
    return regions


def _full_region(regions: list[PlotRegion], pt_variable: str | None) -> PlotRegion | None:
    """Create an inclusive pT region spanning all configured regions.

    Parameters
    ----------
    regions : list[PlotRegion]
        Base plotting regions.
    pt_variable : str | None
        Configured pT variable, if available.

    Returns
    -------
    PlotRegion | None
        Inclusive region from the minimum lower bound to the maximum upper bound,
        or ``None`` if no pT-bounded regions are available.
    """
    ranges = [region.pt_range for region in regions if region.pt_range is not None]
    if not ranges or pt_variable is None:
        return None
    pt_range = (min(r[0] for r in ranges), max(r[1] for r in ranges))
    return PlotRegion("full", _cuts_from_pt_range(pt_variable, pt_range), pt_range)


def _stitching_regions(regions: list[PlotRegion], pt_variable: str | None) -> list[PlotRegion]:
    """Create pT-only windows around adjacent region boundaries.

    Parameters
    ----------
    regions : list[PlotRegion]
        Base pT regions ordered or unordered.
    pt_variable : str | None
        Configured pT variable, if available.

    Returns
    -------
    list[PlotRegion]
        Stitching windows around touching pT-region boundaries. For MeV-scale
        pT values the window half-width is 100 GeV.
    """
    if pt_variable is None:
        return []

    regions_with_pt = sorted(
        [
            (region.name, region.cuts, region.pt_range)
            for region in regions
            if region.pt_range is not None
        ],
        key=lambda region: region[2][0],
    )
    stitching_regions = []
    for idx, (left, right) in enumerate(pairwise(regions_with_pt)):
        boundary = left[2][1]
        if boundary != right[2][0]:
            continue
        half_width = 100_000 if boundary > 10_000 else 100
        pt_range = (boundary - half_width, boundary + half_width)
        name = "stitching" if len(regions_with_pt) == 2 else f"stitching_{idx}"
        stitching_regions.append(
            PlotRegion(name, _cuts_from_pt_range(pt_variable, pt_range), pt_range)
        )
    return stitching_regions


def _load_jets(config: PreprocessingConfig, in_paths: Any, vars_to_load: list[str]) -> Any:
    """Load jet variables for plotting.

    Parameters
    ----------
    config : PreprocessingConfig
        Active preprocessing configuration.
    in_paths : Any
        Input HDF5 file path, glob, or list of paths passed to ``H5Reader``.
    vars_to_load : list[str]
        Jet variables needed for plotting and selections.

    Returns
    -------
    Any
        Structured jet array loaded from the input files.
    """
    return H5Reader(
        fname=in_paths,
        batch_size=config.batch_size,
        jets_name=config.jets_name,
        shuffle=False,
        equal_jets=True,
        vds_dir=config.vds_dir,
    ).load(
        {config.jets_name: list(dict.fromkeys(vars_to_load))},
        num_jets=config.plotting.num_jets_plotting,
    )[config.jets_name]


def make_hist(
    stage: str,
    values_dict: dict,
    flavours: list,
    variable: str,
    out_dir: Path,
    jets_name: str = "jets",
    bins_range: tuple | None = None,
    suffix: str = "",
    out_format_list: tuple[str, ...] | list[str] | None = None,
    selection_cuts: Cuts | None = None,
    atlas_second_tag: str | None = None,
    plotting: PlottingConfig | None = None,
) -> None:
    """Make a flavour-split histogram for one variable.

    The function can overlay multiple input samples via different linestyles and
    multiple flavours via colours. For initial plots, flavour selections come
    from the configured label cuts. For post-resampling plots, flavour selection
    uses the ``flavour_label`` field written during merging.

    Parameters
    ----------
    stage : str
        The stage in which the preprocessing is currently in.
        Mainly used for the ouput name string.
    values_dict : dict
        Dict with the loaded values.
    flavours : list
        List of the flavours that are to be plotted. The list
        needs to contain the Flavour class instances from the
        different flavours.
    variable : str
        Variable that is to be histogrammed and plotted.
    out_dir : Path
        Output directory to which the plots are written.
    jets_name: str, optional
        Name of the jet dataset / the global objects
        by default "jets"
    bins_range : tuple | None, optional
        bins_range argument from from puma.HistogramPlot,
        by default None
    suffix : str, optional
        A string suffix which is added to the plot
        output name, by default "".
    out_format_list : tuple[str, ...] | list[str] | None, optional
        Output formats to save. By default, both "pdf" and "png" are created.
    selection_cuts : Cuts | None, optional
        Additional cuts that are applied before the flavour selection.
    atlas_second_tag : str | None, optional
        Second ATLAS label passed to Puma. If ``None``, only the default
        centre-of-mass energy label is shown.
    plotting : PlottingConfig | None, optional
        Plot labels and style settings. If ``None``, use the defaults.
    """
    from upp.classes.plotting_config import PlottingConfig

    selection_cuts = selection_cuts or Cuts.empty()
    plotting = plotting or PlottingConfig()

    # Setup the histogram
    plot = HistogramPlot(
        ylabel=plotting.ylabel.replace("{jets_name}", jets_name),
        xlabel=plotting.variable_label(variable),
        y_scale=plotting.y_scale,
        figsize=plotting.figsize,
        logy=plotting.logy,
        leg_loc=plotting.legend_location,
        leg_linestyle_loc=plotting.linestyle_legend_location,
        atlas_first_tag=plotting.atlas_first_tag,
        atlas_second_tag=atlas_second_tag or plotting.atlas_second_tag,
        show_xaxis_endpoints=True,
    )

    sample_legend_entries = [
        (plotting.linestyles[counter % len(plotting.linestyles)], label)
        for counter, label in enumerate(values_dict)
        if label
    ]

    for counter, (_values_key, values) in enumerate(values_dict.items()):
        # Loop over the flavours
        for label_value, flavour in enumerate(flavours):
            # Define the cuts that are needed to select the flavours
            if stage == "initial":
                cuts = selection_cuts + flavour.cuts

            else:
                cuts = selection_cuts + Cuts.from_list([f"flavour_label == {label_value}"])

            selected_values = cuts(values).values
            histo_values = _display_values(variable, selected_values[variable])

            # Get the histogram object
            histo = Histogram(
                values=histo_values,
                bins=plotting.bins,
                bins_range=bins_range,
                norm=plotting.norm,
                label=flavour.label if counter == 0 else None,
                colour=flavour.colour,
                linestyle=plotting.linestyles[counter % len(plotting.linestyles)],
                underoverflow=plotting.underoverflow,
            )

            # Add to histogram
            plot.add(histogram=histo)

            # Set bin_edges
            if bins_range is None:
                bins_range = (histo.bin_edges[0], histo.bin_edges[-1])

    # Draw plot
    plot.draw()
    if len(sample_legend_entries) > 1:
        sample_linestyles, sample_labels = zip(*sample_legend_entries, strict=False)
        plot.make_linestyle_legend(
            linestyles=sample_linestyles,
            labels=sample_labels,
            bbox_to_anchor=plotting.linestyle_legend_anchor,
        )

    # Check that the output dir exists
    out_dir.mkdir(parents=True, exist_ok=True)

    # Define output name and path and save it
    fname = f"{stage}_{variable}"

    for iter_out_format in out_format_list or plotting.output_formats:
        out_path = out_dir / f"{fname}{suffix}.{iter_out_format}"
        plot.savefig(out_path)
        log.info(f"Saved plot to {out_path}")


def _plot_initial(config: PreprocessingConfig) -> None:
    """Plot initial distributions per configured sample and region.

    Parameters
    ----------
    config : PreprocessingConfig
        Active preprocessing configuration.
    """
    pt_var = _pt_variable(config)

    for sample, sample_components in config.components.groupby_sample():
        for region, region_components in sample_components.groupby_region():
            selection_cuts = region.cuts
            vars_to_load = list(config.sampl_cfg.vars)
            vars_to_load += selection_cuts.variables
            for flavour in config.components.flavours:
                vars_to_load += flavour.cuts.variables

            values_dict = {
                sample.name: _load_jets(config, list(sample.path), vars_to_load),
            }
            pt_range = _pt_bounds_from_cuts(selection_cuts, pt_var) if pt_var else None

            for variable in config.sampl_cfg.vars:
                bins_range = (
                    _display_range(variable, pt_range)
                    if _is_pt_variable(variable)
                    else _variable_range(config, variable)
                )
                log.info(f"Plotting initial {sample.name} {region.name} {variable}")
                make_hist(
                    stage="initial",
                    values_dict=values_dict,
                    jets_name=config.jets_name,
                    flavours=region_components.flavours,
                    variable=variable,
                    bins_range=bins_range,
                    suffix=_suffix(config.split, sample.name, region.name),
                    selection_cuts=selection_cuts,
                    atlas_second_tag=_atlas_second_tag(
                        sample.name,
                        plotting=config.plotting,
                        num_jets=_plotting_num_jets(config, region_components.num_jets)
                        if config.plotting.show_num_jets
                        else None,
                        resampling_status="Pre Resampling",
                    ),
                    plotting=config.plotting,
                    out_dir=config.out_dir / config.plotting.output_directory,
                )


def _post_resampling_paths(config: PreprocessingConfig, stage: str) -> list[Path]:
    """Return merged output paths used for post-resampling plots.

    Parameters
    ----------
    config : PreprocessingConfig
        Active preprocessing configuration.
    stage : str
        Current preprocessing split or stage name.

    Returns
    -------
    list[Path]
        Paths or glob patterns passed to ``H5Reader``.
    """
    if stage != "test" or config.merge_test_samples:
        return [
            (
                config.out_fname.parent / config.split / f"{config.out_fname.stem}*.h5"
                if config.num_jets_per_output_file is not None
                else config.out_fname
            )
        ]

    paths = [path_append(config.out_fname, sample.name) for sample in config.components.samples]
    if config.num_jets_per_output_file is not None:
        return [path.parent / config.split / f"{path.stem}*.h5" for path in paths]
    return paths


def _plot_post_resampling(config: PreprocessingConfig, stage: str) -> None:
    """Plot merged post-resampling distributions by pT region.

    Parameters
    ----------
    config : PreprocessingConfig
        Active preprocessing configuration.
    stage : str
        Current preprocessing split name.
    """
    pt_var = _pt_variable(config)
    base_regions = _unique_regions(config, pt_var)
    plot_regions = list(base_regions)
    if full_region := _full_region(base_regions, pt_var):
        plot_regions.append(full_region)

    sample_names = [sample.name for sample in config.components.samples]
    atlas_second_tag = _atlas_second_tag(
        *sample_names,
        plotting=config.plotting,
        num_jets=_plotting_num_jets(config, config.components.num_jets)
        if config.plotting.show_num_jets
        else None,
        resampling_status="Post Resampling",
    )

    vars_to_load = list(config.sampl_cfg.vars) + ["flavour_label"]
    for region in plot_regions:
        vars_to_load += region.cuts.variables

    values_dict = {
        "": _load_jets(config, _post_resampling_paths(config, stage), vars_to_load),
    }

    for variable in config.sampl_cfg.vars:
        for region in plot_regions:
            bins_range = (
                _display_range(variable, region.pt_range)
                if _is_pt_variable(variable)
                else _variable_range(config, variable)
            )
            log.info(f"Plotting {stage} {region.name} {variable}")
            make_hist(
                stage=stage,
                values_dict=values_dict,
                jets_name=config.jets_name,
                flavours=config.components.flavours,
                variable=variable,
                bins_range=bins_range,
                suffix=_suffix(region.name),
                selection_cuts=region.cuts,
                atlas_second_tag=atlas_second_tag,
                plotting=config.plotting,
                out_dir=config.out_dir / config.plotting.output_directory,
            )

        if _is_pt_variable(variable):
            for stitching_region in _stitching_regions(base_regions, pt_var):
                log.info(f"Plotting {stage} {stitching_region.name} {variable}")
                make_hist(
                    stage=stage,
                    values_dict=values_dict,
                    jets_name=config.jets_name,
                    flavours=config.components.flavours,
                    variable=variable,
                    bins_range=_display_range(variable, stitching_region.pt_range),
                    suffix=_suffix(stitching_region.name),
                    selection_cuts=stitching_region.cuts,
                    atlas_second_tag=atlas_second_tag,
                    plotting=config.plotting,
                    out_dir=config.out_dir / config.plotting.output_directory,
                )


def plot_resampling_dists(config: PreprocessingConfig, stage: str) -> None:
    """Plot configured resampling variables before or after resampling.

    Parameters
    ----------
    config : PreprocessingConfig
        Active preprocessing configuration.
    stage : str
        ``"initial"`` for pre-resampling sample/region plots, otherwise the
        current split name for post-resampling region, full-range, and stitching
        plots.
    """
    # Nothing to plot when no resampling variables are configured (e.g. resampling skipped)
    if config.sampl_cfg is None or not config.sampl_cfg.vars:
        log.info("No resampling variables configured - skipping resampling plots.")
        return

    log.info("Plotting resampling variable distributions...")
    if stage == "initial":
        _plot_initial(config)
    else:
        _plot_post_resampling(config, stage)
