"""End-to-end test of the benchmark harness on a tiny config."""

import json

import yaml

from stemdenoise.benchmark import _condition_seed, _resolve_preset, run_benchmark
from stemdenoise.net import ResUNet, save_checkpoint
from stemdenoise.sim import preset

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


def test_resolve_preset_string_and_inline():
    name, spec = _resolve_preset("hexagonal")
    assert name == "hexagonal"
    assert spec == preset("hexagonal")
    name, spec = _resolve_preset({"name": "hex_dense", "kind": "hexagonal", "spacing_px": 9.0})
    assert name == "hex_dense"
    assert spec.kind == "hexagonal"
    assert spec.spacing_px == 9.0


def test_benchmark_f1_tuning_and_cnn_path(tmp_path):
    ckpt = str(tmp_path / "tiny.pt")
    save_checkpoint(ResUNet(base=8), ckpt, meta={"mode": "supervised"})
    cfg = dict(
        TINY,
        name="tiny2",
        presets=[{"name": "hex_dense", "kind": "hexagonal", "spacing_px": 9.0}],
        methods=[
            {"name": "gaussian", "tune": True, "tune_metric": "f1"},
            {"name": "cnn_tiny", "checkpoint": ckpt},
        ],
    )
    cfg_path = tmp_path / "tiny2.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    result = run_benchmark(cfg_path, out_path=tmp_path / "tiny2.json")
    rows = result["results"]
    assert {r["method"] for r in rows} == {"gaussian", "cnn_tiny"}
    assert all(r["preset"] == "hex_dense" for r in rows)
    gauss = next(r for r in rows if r["method"] == "gaussian")
    assert gauss["tune_metric"] == "f1"
    cnn = next(r for r in rows if r["method"] == "cnn_tiny")
    # The tiny untrained net is an identity, so its scores equal raw's;
    # the point is that the checkpoint path executes end to end.
    assert 0.0 <= cnn["f1"] <= 1.0


def test_benchmark_reproducible(tmp_path):
    cfg_path = tmp_path / "tiny.yaml"
    cfg_path.write_text(yaml.safe_dump(TINY))
    r1 = run_benchmark(cfg_path, out_path=tmp_path / "a.json")
    r2 = run_benchmark(cfg_path, out_path=tmp_path / "b.json")
    assert r1["results"] == r2["results"]
