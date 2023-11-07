from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass


@dataclass(frozen=True)
class VariableConfig:
    variables: dict[str, dict[str, list[str]]]
    jets_name: str = "jets"
    keep_all: bool = False

    def __post_init__(self):
        for track_vars in self.tracks.values():
            track_vars["inputs"] = list(dict.fromkeys(track_vars["inputs"] + ["valid"]))

    def combined(self):
        combined = {}
        for name in self.variables:
            if self.keep_all:
                combined[name] = None
            else:
                combined[name] = self[name]["inputs"] + self[name].get("labels", [])
        return combined

    @property
    def jets(self):
        return self[self.jets_name]

    @property
    def tracks(self):
        return {name: var for name, var in self.variables.items() if name != self.jets_name}

    def add_jet_vars(self, variables: list[str], kind: str = "inputs") -> VariableConfig:
        """Return a new VariableConfig instance."""
        vc = VariableConfig(deepcopy(self.variables), self.jets_name, self.keep_all)
        vc.jets[kind] = list(dict.fromkeys(vc.jets[kind] + variables))
        return vc

    def items(self):
        return self.variables.items()

    def keys(self):
        return self.variables.keys()

    def __iter__(self):
        yield from self.variables.keys()

    def __getitem__(self, key):
        return self.variables[key]
