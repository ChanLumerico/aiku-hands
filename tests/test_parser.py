"""Parser tests on measured lerobot 0.7.0 stdout (tests/fixtures/lerobot_smoke_stdout.log)."""

from pathlib import Path

import pytest

from aiku_hands.logging.lerobot_log_parser import classify, format_big_number, parse_number, split_stream

FIX = Path(__file__).parent / "fixtures" / "lerobot_smoke_stdout.log"
LINE = ("INFO 2026-11-27 01:45:31 ot_train.py:894 step:20 smpl:160 ep:0 epch:0.00 loss:25.027 "
        "grdn:444.021 lr:1.0e-05 data_s:0.149 prep_s:0.019 updt_s:0.533 step_s:0.701 smp/s:11 "
        "l1_loss:0.827 kld_loss:2.420")


def test_measured_line():
    kind, p = classify(LINE)
    assert kind == "train"
    assert p["step_reported"] == 20
    assert p["samples_reported"] == 160
    assert p["episodes_seen"] == 0
    assert p["epoch"] == 0.0
    assert p["loss"] == pytest.approx(25.027)
    assert p["grad_norm"] == pytest.approx(444.021)
    assert p["lr"] == pytest.approx(1e-5)
    assert p["update_s"] == pytest.approx(0.533)
    assert p["samples_per_s"] == 11
    assert p["l1_loss"] == pytest.approx(0.827)
    assert p["kld_loss"] == pytest.approx(2.420)
    assert p["log_ts"] == "2026-11-27T01:45:31+09:00"


def test_tqdm_glued_line():
    glued = "Training:  10%|█         | 20/200 [00:14<01:01,  2.93step/s]" + LINE
    kind, p = classify(glued)
    assert kind == "train" and p["step_reported"] == 20 and p["loss"] == pytest.approx(25.027)


def test_carriage_return_split():
    text = "Training:   0%|  | 0/200 [00:00<?, ?step/s]\r" + LINE + "\nnext"
    parts = split_stream(text)
    assert parts[1] == LINE and len(parts) == 3


def test_big_number_suffix_kept_raw():
    kind, p = classify(LINE.replace("smpl:160", "smpl:1K").replace("step:20", "step:14K"))
    assert p["samples_reported"] == 1000 and p["samples_reported_raw"] == "1K"
    assert p["step_reported"] == 14000 and p["step_reported_raw"] == "14K"


def test_format_big_number_replica():
    assert format_big_number(999) == "999"
    assert format_big_number(1000) == "1K"
    assert format_big_number(14500) == "14K"  # banker's rounding in f-string .0f
    assert format_big_number(15500) == "16K"
    assert format_big_number(70000) == "70K"


@pytest.mark.parametrize("s,v", [("160", 160), ("1K", 1000), ("2M", 2_000_000), ("0.00", 0.0), ("1.0e-05", 1e-5)])
def test_parse_number(s, v):
    assert parse_number(s) == pytest.approx(v)


def test_events_and_noise():
    assert classify("INFO 2026-11-27 01:46:32 t_train.py:1026 End of training")[0] == "end"
    k, p = classify("INFO 2026-11-27 03:00:00 ot_train.py:936 Checkpoint policy after step 14000")
    assert k == "ckpt" and p["step"] == 14000
    k, p = classify("INFO 2026-11-27 03:00:00 ot_train.py:928 step 500: eval_loss=0.1234")
    assert k == "eval_loss" and p["eval_loss"] == pytest.approx(0.1234)
    assert classify("objc[69716]: Class AVFFrameReceiver is implemented in both ...")[0] == "ignore"
    assert classify("W1127 01:45:19.196000 69716 torch/distributed/elastic/multiprocessing/redirects.py:35] "
                    "NOTE: Redirects are currently not supported in MacOs.")[0] == "ignore"
    assert classify("Training:  50%|█████     | 100/200 [00:41<00:34,  2.91step/s]")[0] == "ignore"
    k, p = classify("INFO 2026-11-27 01:45:17 ot_train.py:799 num_learnable_params=51597190 (52M)")
    assert k == "meta" and p["num_learnable_params"] == 51597190


def test_malformed_metrics_line_raises():
    with pytest.raises(ValueError):
        classify("INFO 2026-11-27 01:45:31 ot_train.py:894 step:20 loss:abc")


def test_fixture_has_10_train_lines():
    kinds = [classify(l)[0] for l in split_stream(FIX.read_text())]
    assert kinds.count("train") == 10
    assert kinds.count("end") == 1
