"""Stand-in for sim/act_policy_server.py used by `sim/eval.py run --mock` (no lerobot, no GPU)."""
import argparse
import time

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt")
ap.add_argument("--device")
ap.add_argument("--port", type=int)
a, _ = ap.parse_known_args()
time.sleep(1.0)  # "model load"
print(f"loaded mock ckpt={a.ckpt}", flush=True)
print(f"listening on 127.0.0.1:{a.port}", flush=True)
while True:
    time.sleep(1.0)
