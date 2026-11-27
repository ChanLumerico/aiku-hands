"""sim/eval.py pipeline test with the mock server/rollout (no Isaac, no lerobot)."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sim"))
import eval as sim_eval  # noqa: E402


def test_wilson_matches_reference():
    lo, hi = sim_eval.wilson(3, 10)
    assert abs(lo - 0.108) < 1e-3 and abs(hi - 0.603) < 1e-3


def test_cond_name():
    assert sim_eval.cond_name({"ckpt": 70000, "horizon": 25}) == "ck70k_h25"
    assert sim_eval.cond_name({"ckpt": 12345, "horizon": 100}) == "ck12345_h100"


def test_mock_run_end_to_end(tmp_path):
    cfg = tmp_path / "t.json"
    cfg.write_text(json.dumps({"episodes": 2, "episode_length_s": 5,
                               "conditions": [{"ckpt": 70000, "horizon": 100}, {"ckpt": 70000, "horizon": 50}]}))
    r = subprocess.run([sys.executable, str(REPO / "sim" / "eval.py"), "run", str(cfg), "--mock"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    rd = (REPO / "runs" / "sim_eval" / "latest").resolve()
    try:
        for c in ("ck70k_h100", "ck70k_h50"):
            recs = [json.loads(x) for x in (rd / "rollouts" / f"{c}.jsonl").read_text().splitlines()]
            assert len(recs) == 2
        evs = [json.loads(x)["ev"] for x in (rd / "progress.jsonl").read_text().splitlines()]
        assert evs[0] == "run_start" and evs[-1] == "run_end" and evs.count("cond_done") == 2
    finally:
        shutil.rmtree(rd)
        (REPO / "runs" / "sim_eval" / "latest").unlink()
