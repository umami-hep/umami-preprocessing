from __future__ import annotations

from pathlib import Path

from upp.utils.list_components import main

CONFIG = Path(__file__).parents[3] / "upp/configs/test.yaml"


def test_list_components(capsys):
    main(["--config", str(CONFIG)])
    rows = [line.split("\t") for line in capsys.readouterr().out.splitlines()]
    assert len(rows) == 6
    assert all(len(row) == 4 for row in rows)
    assert ["lowpt", "ttbar", "bjets", "lowpt_ttbar_bjets"] in rows
    assert ["highpt", "zprime", "cjets", "highpt_zprime_cjets"] in rows


def test_list_components_regions(capsys):
    main(["--config", str(CONFIG), "--regions"])
    assert capsys.readouterr().out.splitlines() == ["lowpt", "highpt"]
