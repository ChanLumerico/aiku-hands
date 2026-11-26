"""Step reconstruction + end-to-end ingest on the measured smoke log."""

from pathlib import Path

from aiku_hands.logging.ingest import LineBuffer, TrainLogIngestor
from aiku_hands.logging.jsonl import JsonlWriter, read_jsonl

FIX = Path(__file__).parent / "fixtures" / "lerobot_smoke_stdout.log"
TMPL = ("INFO {ts} ot_train.py:894 step:{s} smpl:{n} ep:1 epch:0.10 loss:1.0 grdn:9.0 "
        "lr:1.0e-05 data_s:0.003 prep_s:0.000 updt_s:0.337 step_s:0.340 smp/s:24 l1_loss:0.1 kld_loss:0.01")


def _ts(step, step_s=0.34):
    from datetime import datetime, timedelta
    return (datetime(2026, 11, 27, 2, 0, 0) + timedelta(seconds=round(step * step_s))).strftime("%Y-%m-%d %H:%M:%S")


def _ing(tmp_path, **kw):
    ws = [JsonlWriter(tmp_path / f, "E9", "r", fsync=False) for f in ("t.jsonl", "e.jsonl", "v.jsonl")]
    return TrainLogIngestor(*ws, **kw), ws


def _fmt(x):
    from aiku_hands.logging.lerobot_log_parser import format_big_number
    return format_big_number(x)


def test_step_reconstruction_log_freq_500(tmp_path):
    ing, ws = _ing(tmp_path, batch_size=8, log_freq=500, total_steps=70000, use_log_time=True)
    for i, step in enumerate(range(500, 5001, 500)):
        ing.feed(TMPL.format(ts=_ts(step), s=_fmt(step), n=_fmt(step * 8)))
    ing.finalize(0)
    [w.close() for w in ws]
    rows = list(read_jsonl(tmp_path / "t.jsonl"))
    assert [r["step"] for r in rows] == list(range(500, 5001, 500))
    assert all(r["step_exact"] for r in rows)
    assert rows[-1]["samples"] == 5000 * 8
    assert rows[1]["wall_s"] == 170.0  # 500 steps x 0.34 s


def test_lost_line_is_skipped_over(tmp_path):
    ing, ws = _ing(tmp_path, batch_size=8, log_freq=500)
    for i, step in enumerate([500, 1000, 2000, 2500]):  # 1500 missing
        ing.feed(TMPL.format(ts=_ts(step), s=_fmt(step), n=_fmt(step * 8)))
    [w.close() for w in ws]
    rows = list(read_jsonl(tmp_path / "t.jsonl"))
    assert [r["step"] for r in rows] == [500, 1000, 2000, 2500]


def test_parse_error_goes_to_events(tmp_path):
    ing, ws = _ing(tmp_path, batch_size=8, log_freq=500)
    ing.feed("INFO 2026-11-27 01:45:31 ot_train.py:894 step:20 loss:abc")
    [w.close() for w in ws]
    ev = list(read_jsonl(tmp_path / "e.jsonl"))
    assert ev[0]["type"] == "parse_error" and "loss:abc" in ev[0]["raw"]


def test_ckpt_event_and_size(tmp_path):
    run = tmp_path / "run"
    ck = run / "checkpoints" / "014000"
    ck.mkdir(parents=True)
    (ck / "model.bin").write_bytes(b"x" * 2**20)
    ing, ws = _ing(tmp_path, batch_size=8, log_freq=500, total_steps=70000, run_dir=run)
    ing.feed("INFO 2026-11-27 03:00:00 ot_train.py:936 Checkpoint policy after step 14000")
    ing.finalize(0)
    [w.close() for w in ws]
    ev = [e for e in read_jsonl(tmp_path / "e.jsonl") if e["type"] == "ckpt_saved"]
    assert ev[0]["step"] == 14000 and ev[0]["size_mb"] == 1.0 and ev[0]["path"].endswith("014000")


def test_smoke_fixture_end_to_end(tmp_path):
    ing, ws = _ing(tmp_path, batch_size=8, log_freq=20, total_steps=200, use_log_time=True)
    lb = LineBuffer()
    for line in lb.push(FIX.read_text()) + lb.flush():
        ing.feed(line)
    ing.finalize(0)
    [w.close() for w in ws]
    rows = list(read_jsonl(tmp_path / "t.jsonl"))
    assert [r["step"] for r in rows] == list(range(20, 201, 20))
    assert rows[-1]["samples"] == 1600  # lerobot prints 'smpl:2K'
    assert rows[0]["loss"] == 25.027 and rows[-1]["loss"] == 3.545
    assert ing.counts["parse_error"] == 0 and ing.saw_end


def test_linebuffer_partial_chunks():
    lb = LineBuffer()
    assert lb.push("abc") == []
    assert lb.push("def\rgh") == ["abcdef"]
    assert lb.push("\n") == ["gh"]
