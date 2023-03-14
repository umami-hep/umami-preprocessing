from __future__ import annotations

import numpy as np


class ResamplingConfig:
    def __init__(self, config):
        self.target = config["resampling"]["target"]
        self.sampling_fraction = config["resampling"]["sampling_fraction"]
        self.method = config["resampling"].get("method")

        self.vars = list(config["resampling"]["variables"].keys())

        self.bins = {}
        for variable, var_config in config["resampling"]["variables"].items():
            self.bins[variable] = var_config["bins"]

    def get_bins_x(self, bins_x):
        flat_bins = []
        for i, sub_bins_x in enumerate(bins_x):
            start, stop, nbins = sub_bins_x
            b = np.linspace(start, stop, nbins + 1)
            if i > 0:
                b = b[1:]
            flat_bins.append(b)
        return np.concatenate(flat_bins)

    @property
    def flat_bins(self):
        return [self.get_bins_x(bins_x) for bins_x in self.bins.values()]
