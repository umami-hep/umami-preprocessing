from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ResamplingConfig:
    variables: dict
    target: str
    sampling_fraction: float = 1.0
    method: str | None = None
    upscale_pdf: int | None = None

    @property
    def vars(self):
        return list(self.variables.keys())

    @property
    def bins(self):
        return {v: vc["bins"] for v, vc in self.variables.items()}

    def get_bins_x(self, bins_x, upscale=1):
        flat_bins = []
        for i, sub_bins_x in enumerate(bins_x):
            start, stop, nbins = sub_bins_x
            b = np.linspace(start, stop, nbins * upscale + 1)
            if i > 0:
                b = b[1:]
            flat_bins.append(b)
        return np.concatenate(flat_bins)

    @property
    def flat_bins(self):
        return [self.get_bins_x(bins_x) for bins_x in self.bins.values()]
