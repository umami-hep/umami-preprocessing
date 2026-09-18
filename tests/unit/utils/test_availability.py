from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from ftag import Cuts

import upp.utils.availability as av


def _objects(pt):
    array = np.zeros(len(pt), dtype=[("pt", "f4")])
    array["pt"] = pt
    return array


class _SingleReader:
    def __init__(self, pt, num_global_objects):
        self.objects = _objects(pt)
        self.num_global_objects = num_global_objects

    def stream(self, _variables, _num):
        yield {"jets": self.objects}


class _Reader:
    global_objects_name = "jets"

    def __init__(self, readers, equal_global_objects=False):
        self.readers = readers
        self.equal_global_objects = equal_global_objects

    @property
    def num_global_objects(self):
        return sum(r.num_global_objects for r in self.readers)

    def load(self, _variables, _num):
        return {"jets": np.concatenate([r.objects for r in self.readers])}


def _component(reader, name="lowpt_ttbar_bjets", pattern=("data1.h5",)):
    return SimpleNamespace(
        name=name,
        reader=reader,
        equal_global_objects=reader.equal_global_objects,
        sample=SimpleNamespace(name="ttbar", pattern=pattern),
        region=SimpleNamespace(name="lowpt"),
        flavour=SimpleNamespace(name="bjets"),
        is_target=lambda target: target == "bjets",
    )


def test_estimate_availability():
    reader = _Reader([_SingleReader(np.arange(100), 1000)])
    split_cuts = {
        "train": Cuts.from_list([["pt", ">", 49]]),
        "val": Cuts.from_list([["pt", ">", 74]]),
    }
    estimates = av.estimate_availability(_component(reader), split_cuts, None)
    assert estimates == {"train": 495, "val": 247}


def test_estimate_availability_equal_global_objects():
    passing_25 = np.concatenate([np.zeros(75), np.full(25, 100)])
    reader = _Reader(
        [_SingleReader(np.arange(100), 1000), _SingleReader(passing_25, 1000)],
        equal_global_objects=True,
    )
    split_cuts = {"train": Cuts.from_list([["pt", ">", 49]])}
    # the second sample has half as many objects passing the cuts, and sets the estimate
    assert av.estimate_availability(_component(reader), split_cuts, None) == {"train": 495}


def test_solve_counts_auto_ratios():
    available = {
        "train": {
            ("lowpt", "bjets"): 1000,
            ("lowpt", "cjets"): 500,
            ("highpt", "bjets"): 1000,
            ("highpt", "cjets"): 1000,
        }
    }
    counts = av.solve_counts(available, {"lowpt": 2, "highpt": 1})["train"]

    # bjets are limited by lowpt (1000/2), cjets by lowpt as well (500/2)
    assert counts == {
        ("lowpt", "bjets"): 1000,
        ("lowpt", "cjets"): 500,
        ("highpt", "bjets"): 500,
        ("highpt", "cjets"): 250,
    }


def test_solve_counts_keeps_exact_ratios():
    available = {
        "train": {
            ("lowpt", "bjets"): 220_586_493,
            ("lowpt", "cjets"): 50_743_576,
            ("highpt", "bjets"): 166_365_630,
            ("highpt", "cjets"): 171_046_924,
        }
    }
    counts = av.solve_counts(available, {"lowpt": 2, "highpt": 1})["train"]

    for name in ("bjets", "cjets"):
        assert counts[("lowpt", name)] == 2 * counts[("highpt", name)]
        assert counts[("lowpt", name)] <= available["train"][("lowpt", name)]

    lowpt_ratio = counts[("lowpt", "bjets")] / counts[("lowpt", "cjets")]
    highpt_ratio = counts[("highpt", "bjets")] / counts[("highpt", "cjets")]
    assert lowpt_ratio == highpt_ratio


def test_solve_counts_fixed_ratios():
    available = {
        "train": {
            ("lowpt", "bjets"): 1000,
            ("lowpt", "cjets"): 9000,
            ("highpt", "bjets"): 5000,
            ("highpt", "cjets"): 5000,
        }
    }
    counts = av.solve_counts(available, {"lowpt": 1, "highpt": 1}, {"bjets": 1, "cjets": 3})[
        "train"
    ]

    # bjets are the bottleneck, so the requested 1:3 fills them up completely
    assert counts[("lowpt", "bjets")] == 1000
    assert counts[("lowpt", "cjets")] == 3000
    assert all(counts[key] <= available["train"][key] for key in counts)


def test_sampling_factor():
    config = SimpleNamespace(
        skip_resampling=False,
        sampl_cfg=SimpleNamespace(target="bjets", sampling_fraction=0.5),
    )
    target = _component(_Reader([]))
    other = _component(_Reader([]))
    other.is_target = lambda _target: False

    assert av.sampling_factor(config, target) == 1.0
    assert av.sampling_factor(config, other) == 0.5

    config.sampl_cfg.sampling_fraction = "auto"
    assert av.sampling_factor(config, other) == pytest.approx(1 / 1.1)

    config.skip_resampling = True
    assert av.sampling_factor(config, other) == 1.0


def test_region_weights():
    config = SimpleNamespace(
        config={
            "components": [
                {"region": {"name": "lowpt"}, "sample": {"sample_weight": 2}},
                {"region": {"name": "lowpt"}, "sample": {"sample_weight": 2}},
                {"region": {"name": "highpt"}, "sample": {}},
            ]
        }
    )
    assert av.region_weights(config) == {"lowpt": 2, "highpt": 1}


def test_region_weights_inconsistent():
    config = SimpleNamespace(
        config={
            "components": [
                {"region": {"name": "lowpt"}, "sample": {"sample_weight": 2}},
                {"region": {"name": "lowpt"}, "sample": {"sample_weight": 3}},
            ]
        }
    )
    with pytest.raises(ValueError, match="different sample_weight"):
        av.region_weights(config)


def test_target_ratios():
    config = SimpleNamespace(config={})
    assert av.target_ratios(config) is None

    config.config = {"auto_counts": {"class_ratios": {"bjets": 1, "cjets": 2}}}
    assert av.target_ratios(config) == {"bjets": 1, "cjets": 2}


def test_cache_key_changes_with_cuts():
    component = _component(_Reader([]))
    cuts = Cuts.from_list([["pt", ">", 20]])
    other = Cuts.from_list([["pt", ">", 30]])

    assert av.cache_key(component, cuts, 100) == av.cache_key(component, cuts, 100)
    assert av.cache_key(component, cuts, 100) != av.cache_key(component, other, 100)
    assert av.cache_key(component, cuts, 100) != av.cache_key(component, cuts, 200)


def test_cache_roundtrip(tmp_path):
    config = SimpleNamespace(
        config_path=tmp_path / "config.yaml",
        out_dir=tmp_path / "out",
        num_global_objects_estimate_available=1000,
    )
    components = {"lowpt_ttbar_bjets": {"train": {"available": 10, "key": "abc"}}}

    assert av.read_cache(av.cache_path(config)) == {}
    av.write_cache(av.cache_path(config), config, components)

    assert av.read_cache(av.cache_path(config)) == components
    written = yaml.safe_load((tmp_path / "out" / av.CACHE_FNAME).read_text())
    assert written["config"] == str(tmp_path / "config.yaml")


def test_parse_args(tmp_path):
    config = tmp_path / "cfg.yaml"
    config.write_text("")
    args = av.parse_args(["--config", str(config), "--splits", "train,val", "--force"])
    assert args.splits == ("train", "val")
    assert args.force is True


def test_solve_counts_reproduces_central_config():
    # availability and counts of the 260826 central config, which was solved by hand
    available = {
        "ghostbjets": (220_586_493, 166_365_630),
        "ghostcjets": (50_743_576, 171_046_924),
        "ghostsjets": (40_055_330, 65_906_199),
        "ghostudjets": (102_657_253, 122_176_309),
        "ghostgjets": (120_698_336, 136_594_381),
        "ghosttaujets": (21_217_698, 45_486_757),
    }
    expected = {
        "ghostbjets": (220_586_492, 110_293_246),
        "ghostcjets": (50_743_576, 25_371_788),
        "ghostsjets": (40_055_330, 20_027_665),
        "ghostudjets": (102_657_252, 51_328_626),
        "ghostgjets": (120_698_336, 60_349_168),
        "ghosttaujets": (21_217_698, 10_608_849),
    }
    counts = av.solve_counts(
        {
            "train": {
                (region, name): available[name][i]
                for name in available
                for i, region in enumerate(("lowpt", "highpt"))
            }
        },
        {"lowpt": 2, "highpt": 1},
    )["train"]

    for name, (lowpt, highpt) in expected.items():
        assert counts[("lowpt", name)] == lowpt
        assert counts[("highpt", name)] == highpt
