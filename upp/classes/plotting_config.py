from __future__ import annotations

from dataclasses import dataclass, field


def _default_variable_labels() -> dict[str, str]:
    return {
        "pt": "Jet $p_\\mathrm{T}$ [GeV]",
        "eta": "Jet $|\\eta|$",
        "mass": "Jet Mass [GeV]",
    }


def _default_sample_labels() -> dict[str, str]:
    return {
        "ttbar": "$t\\bar{t}$",
        "zprime": "$Z'$",
    }


@dataclass
class PlottingConfig:
    r"""
    Options for the preprocessing resampling distribution plots.

    These options are specified in the config file under the `plotting:` key.
    Any omitted option uses the default defined by this class.

    Attributes
    ----------
    num_jets_plotting : int | None, optional
        Number of jets loaded for plotting. If not set, use the global
        `num_jets_estimate_plotting` value. By default None.
    variable_labels : dict[str, str], optional
        Display labels for plotted variables. Keys are matched case-insensitively
        against variable names, with the longest matching key taking precedence.
        User-provided labels are merged with the default pT, eta, and mass labels.
    sample_labels : dict[str, str], optional
        Display labels for input samples. User-provided labels are merged with the
        default ttbar and zprime labels.
    ylabel : str, optional
        Label for the y-axis. The `{jets_name}` placeholder is replaced with the
        configured jet dataset name. By default "Normalised Number of {jets_name}".
    atlas_first_tag : str, optional
        First ATLAS plot label. By default "Simulation Internal".
    atlas_second_tag : str, optional
        Second ATLAS plot label. By default "$\\sqrt{s} = 13/13.6\\,\\mathrm{TeV}$".
    output_formats : list[str], optional
        File formats in which each plot is saved. By default `["pdf", "png"]`.
    linestyles : list[str], optional
        Linestyles used to distinguish input samples. By default
        `["-", "--", "-.", ":"]`.
    bins : int, optional
        Number of histogram bins. By default 50.
    norm : bool, optional
        Normalise each histogram before plotting. By default True.
    underoverflow : bool, optional
        Include underflow and overflow values in the edge bins. By default True.
    y_scale : float, optional
        Scale factor applied to the automatically determined y-axis range.
        By default 1.5.
    figsize : list[float], optional
        Figure width and height. By default `[6, 4]`.
    logy : bool, optional
        Use a logarithmic y-axis. By default True.
    legend_location : str, optional
        Location of the flavour legend. By default "upper right".
    linestyle_legend_location : str, optional
        Location of the sample-linestyle legend. By default "upper center".
    linestyle_legend_anchor : list[float], optional
        Anchor position of the sample-linestyle legend. By default `[0.55, 1]`.
    output_directory : str, optional
        Plot directory relative to the preprocessing output directory.
        By default "plots".
    """

    num_jets_plotting: int | None = None
    variable_labels: dict[str, str] = field(default_factory=_default_variable_labels)
    sample_labels: dict[str, str] = field(default_factory=_default_sample_labels)
    ylabel: str = "Normalised Number of {jets_name}"
    atlas_first_tag: str = "Simulation Internal"
    atlas_second_tag: str = "$\\sqrt{s} = 13/13.6\\,\\mathrm{TeV}$"
    output_formats: list[str] = field(default_factory=lambda: ["pdf", "png"])
    linestyles: list[str] = field(default_factory=lambda: ["-", "--", "-.", ":"])
    bins: int = 50
    norm: bool = True
    underoverflow: bool = True
    y_scale: float = 1.5
    figsize: list[float] = field(default_factory=lambda: [6, 4])
    logy: bool = True
    legend_location: str = "upper right"
    linestyle_legend_location: str = "upper center"
    linestyle_legend_anchor: list[float] = field(default_factory=lambda: [0.55, 1])
    output_directory: str = "plots"

    def __post_init__(self) -> None:
        self.variable_labels = {**_default_variable_labels(), **self.variable_labels}
        self.sample_labels = {**_default_sample_labels(), **self.sample_labels}
        if self.num_jets_plotting is not None and self.num_jets_plotting <= 0:
            raise ValueError("plotting.num_jets_plotting must be a positive integer or None")
        if not self.output_formats:
            raise ValueError("plotting.output_formats must contain at least one format")
        if not self.linestyles:
            raise ValueError("plotting.linestyles must contain at least one linestyle")

    def variable_label(self, variable: str) -> str:
        """Return the configured display label for a variable."""
        variable_lower = variable.lower()
        for name, label in sorted(
            self.variable_labels.items(), key=lambda item: len(item[0]), reverse=True
        ):
            if name.lower() in variable_lower:
                return label
        return variable

    def sample_label(self, sample: str) -> str:
        """Return the configured display label for a sample."""
        return self.sample_labels.get(sample, sample)
