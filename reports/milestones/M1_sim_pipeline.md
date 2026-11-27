# M1 · sim pipeline 동작 (W6)
> TL;DR (EN): Tracking doc for M1 — train ACT on sim data, then roll it out in LeIsaac with the 9-grid protocol and
> log `rollouts_sim.jsonl`. Training side is verified (E000). The sim-rollout side has not started.

| 구성 요소 | 상태 | 근거 |
|---|---|---|
| sim 데이터 | ✅ cohab orange 100 ep | docs/03_data.md |
| 학습 (Mac Studio) | ✅ E001 done (70k, 6.66 h) | E001 |
| 로깅 | ✅ v0 | docs/05_logging_spec.md |
| checkpoint → HANDS 전송 | ⬜ `sync_ckpt_to_hands.sh` 준비됨, ssh 미확인 | |
| sim rollout 9-grid × 50 | ⬜ 좌표·스크립트 미정 | configs/eval/ |
| $S_{\text{sim}}$ ± Wilson CI | ⬜ | |
