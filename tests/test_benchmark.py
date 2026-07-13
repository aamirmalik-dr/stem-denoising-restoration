"""End-to-end test of the benchmark harness on a tiny config."""

import json

import yaml

from stemdenoise.benchmark import _condition_seed, run_benchmark

TINY = {
    "name": "tiny",
    "seed": 7,
    "presets": ["hexagonal"],
    "doses": [10],
    "n_eval_fields": 2,
    "n_tune_fields": 1,
    "field_size": 96,
    "tolerance_px": 3.0,
    "methods": [{"name": "raw"}, {"name": "gaussian", "tune": True}],
}


def test_condition_seed_stable():
    a = _condition_seed(7, "hexagonal", 10.0, "eval")
    b = _condition_seed(7, "hexagonal", 10.0, "eval")
    assert a == b
    assert a != _condition_seed(7, "hexagonal", 10.0, "tune")


def test_tiny_benchmark(tmp_path):
    cfg_path = tmp_path / "tiny.yaml"
    cfg_path.write_text(yaml.safe_dump(TINY))
    out_path = tmp_path / "tiny.json"
    result = run_benchmark(cfg_path, out_path=out_path)
    assert out_path.exists()
    saved = json.loads(out_path.read_text())
    assert saved["results"] == result["results"]
    rows = result["results"]
    assert len(rows) == 2
    gauss = next(r for r in rows if r["method"] == "gaussian")
    raw = next(r for r in rows if r["method"] == "raw")
    assert "tuned_param" in gauss
    assert gauss["psnr"] > raw["psnr"]


def test_benchmark_reproducible(tmp_path):
    cfg_path = tmp_path / "tiny.yaml"
    cfg_path.write_text(yaml.safe_dump(TINY))
    r1 = run_benchmark(cfg_path, out_path=tmp_path / "a.json")
    r2 = run_benchmark(cfg_path, out_path=tmp_path / "b.json")
    assert r1["results"] == r2["results"]
