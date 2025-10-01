from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ReweightConfig:
    # Number of jets to estimate, if None, use the global num jets estimate
    num_jets_estimate: None | int = None
    merge_num_proc: int = 1  # Number of processes to use for merging
    reweights: list[SingleReweightConfig] = field(default_factory=list)

    def __post_init__(self):
        if self.num_jets_estimate is not None and self.num_jets_estimate <= 0:
            raise ValueError("num_jets_estimate must be a positive integer or None")

        parsed_reweights = []
        for rw in self.reweights:
            parsed_reweights.append(SingleReweightConfig(**rw))
        self.reweights = parsed_reweights


@dataclass
class SingleReweightConfig:
    group: str  # The group our variables in the h5 file are in
    reweight_vars: list[str]  # The variables we want to reweight
    bins: list[np.ndarray]  # The bins we want to use for the reweighting
    class_var: str  # The variable which contains the label we resample over, e.g. flavour
    class_target: int | tuple | str | None = None
    add_overflow: bool = True  # Whether to add overflow bins

    target_hist_func: Callable | None = None
    target_hist_func_name: str | None = None

    # TODO - this is the same as in resampling, maybe can cleanup
    def get_bins_x(self, bins_x, upscale=1):
        flat_bins = []
        for i, sub_bins_x in enumerate(bins_x):
            start, stop, nbins = sub_bins_x
            b = np.linspace(start, stop, nbins * upscale + 1)
            if i > 0:
                b = b[1:]
            flat_bins.append(b)
        if self.add_overflow:
            flat_bins = [np.array([-np.inf])] + flat_bins + [np.array([np.inf])]
        return np.concatenate(flat_bins)

    @property
    def flat_bins(self):
        return [self.get_bins_x(self.bins[k]) for k in self.reweight_vars]

    def __post_init__(self):
        if isinstance(self.class_target, str) and self.class_target not in [
            "mean",
            "min",
            "max",
            "uniform",
        ]:
            raise ValueError("class_target must be either 'mean', 'min', 'max' or an integer")

        if self.target_hist_func is not None and self.target_hist_func_name is None:
            self.target_hist_func_name = self.target_hist_func.__name__

    def __repr__(self):
        target_str = "target_"
        if self.target_hist_func_name is not None:
            target_str += f"{self.target_hist_func_name}_"
        if self.class_target is not None:
            if isinstance(self.class_target, list | tuple):
                target_str += "_".join(map(str, self.class_target))
            else:
                target_str += f"{self.class_target}_{self.class_var}"
        else:
            target_str += "none"
        return f"weight_{self.group}_{'_'.join(self.reweight_vars)}_{target_str}"
