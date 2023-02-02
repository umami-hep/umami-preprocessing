from __future__ import annotations

import operator
from dataclasses import dataclass

import numpy as np

OPERATORS = {
    "==": operator.__eq__,
    ">=": operator.__ge__,
    "<=": operator.__le__,
    ">": operator.__gt__,
    "<": operator.__lt__,
    "%10==": lambda x, y: (x % 10) == y,
    "%10<=": lambda x, y: (x % 10) <= y,
}


@dataclass(frozen=True)
class Cut:
    variable: str
    operator: str
    value: str

    def __call__(self, array):
        return OPERATORS[self.operator](array[self.variable], self.value)


@dataclass(frozen=True)
class Cuts:
    cuts: frozenset[Cut]

    @classmethod
    def from_list(cls, cuts: list) -> Cuts:
        return cls(frozenset(Cut(*cut) for cut in cuts))

    @property
    def variables(self):
        return list({cut.variable for cut in self.cuts})

    def ignore(self, variables):
        return Cuts(frozenset(c for c in self.cuts if c.variable not in variables))

    def __call__(self, array):
        idx = np.arange(len(array))
        for cut in self.cuts:
            keep_idx = cut(array)
            idx = idx[keep_idx]
            array = array[keep_idx]
        return idx, array

    def __len__(self):
        return len(self.cuts)

    def __getitem__(self, index):
        return list(self.cuts)[index]

    def __add__(self, other):
        return Cuts(self.cuts | other.cuts)
