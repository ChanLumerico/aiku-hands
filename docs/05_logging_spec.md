# 05. 로깅 사양 / Logging Spec
> TL;DR (EN): Everything is append-only JSON Lines, one event per line, flushed per line, with common fields
> `schema_version, exp_id, run_id, ts (+09:00), wall_s`. v0 = `scripts/run_train.py` wrapping `lerobot-train`
> (stdout parse + psutil sysmon + manifest). lerobot prints `step` lossily (`14K`), so the exact step is
> reconstructed from `log_freq` and flagged with `step_exact`.

## 1. 원칙 / Principles
- JSON Lines, 한 줄 = 한 이벤트, **append-only**, 매 줄 flush + fsync ([jsonl.py](../src/aiku_hands/logging/jsonl.py))
- 공통 필드: `schema_version` (현재 1), `exp_id`, `run_id`, `ts` (ISO-8601, Asia/Seoul), `wall_s` (run 시작 이후 초)
- 단위가 드러나는 키: `_s`, `_mb`, `_gb`, `_pct`
- NaN/inf는 `null`로 저장 (strict JSON)
- 파싱 실패 줄은 버리지 않음 → `events.jsonl`에 `{"type":"parse_error","raw":...,"error":...}`
  - "metrics 줄처럼 보이는데 파싱 실패"한 경우만 parse_error로 기록함. 일반 INFO 줄은 raw 로그(`logs/`)에만 남김
  - `WARNING`/`ERROR` 줄은 `log_warning` 이벤트로 기록함

## 2. 파일별 스키마 / Per-file schema

### 2.1 `train.jsonl` — lerobot `log_freq`마다 1줄
| key | 출처 | 비고 |
|---|---|---|
| `step` | 복원값 | 아래 §3 |
| `step_exact` | bool | 복원 확신 여부 |
| `samples` | `step × batch_size` | lerobot `smpl`은 `1K` 축약이라 사용 안 함 |
| `epoch` | lerobot `epch` (소수 2자리) | |
| `epoch_exact` | `samples / num_frames` | `dataset.num_frames` 로그 줄이 있을 때만 |
| `episodes_seen` (+`_raw`) | lerobot `ep` | 축약됨 |
| `loss`, `l1_loss`, `kld_loss` | lerobot | log_freq 창 평균. ACT: $\text{loss} = \text{l1} + \text{kl\_weight}\cdot\text{kld}$ (kl_weight=10) |
| `grad_norm` | `grdn` | clip 전 norm |
| `lr` | `lr` | |
| `data_s`, `prep_s`, `update_s`, `step_s` | `data_s`, `prep_s`, `updt_s`, `step_s` | 창 평균 |
| `samples_per_s` | `smp/s` | 정수 반올림 |
| `samples_reported` (+`_raw`) | `smpl` | 원본 보존용 |
| `log_ts` | 로그 헤더 시각 | 1 s 해상도 |
- lerobot이 새 metric을 추가해도 버리지 않고 그대로 기록함

### 2.2 `system.jsonl` — 30 s마다 ([sysmon.py](../src/aiku_hands/logging/sysmon.py))
`pid, n_procs, cpu_pct (프로세스 트리 합, 1코어=100), rss_mb (트리 합), sys_mem_used_pct, swap_used_mb, mps_alloc_mb, mps_driver_mb, disk_free_gb, thermal_state`
- v0은 별도 프로세스이므로 `mps_*`는 `null`임. `torch.mps.current_allocated_memory()`/`driver_allocated_memory()`는 v1(in-process)에서 기록함
- `rss_mb`는 DataLoader worker 포함 합계. unified memory 공유 페이지가 중복으로 세어질 수 있음 (🔮 과대평가 가능)
- `thermal_state`: `pmset -g therm`에 경고가 없으면 `nominal`

### 2.3 `events.jsonl`
| type | 필드 |
|---|---|
| `run_start` | `cmdline`, `manifest`, `raw_log` |
| `run_info` | `total_steps`, `num_frames`, `num_episodes`, `num_learnable_params`, `effective_batch_size`, `n_train_episodes`/`n_eval_episodes` |
| `ckpt_saved` | `step` (정확값), `path`, `size_mb` (다음 로그 줄 시점에 측정) |
| `training_end_logged` | `last_step` |
| `run_end` | `exit_code`, `total_wall_s`, `last_step`, `counts` |
| `error` | `exit_code`, `traceback_tail` (마지막 30줄) |
| `disk_warning` | `disk_free_gb` (< 5 GB) |
| `parse_error`, `log_warning` | `raw` |
| `backfill_start`/`backfill_end` | `source_log`, `source_sha256`, `counts` |

### 2.4 `offline_eval.jsonl`
- (a) lerobot 내장 held-out loss (`--eval_steps`, `--dataset.eval_split`): `{"source":"lerobot_eval_loss","split":"heldout","step":N,"eval_loss":x}`
- (b) checkpoint별 action 오차 (`offline_eval.py`, 미구현):
```json
{"ckpt_step":14000,"split":"heldout","n_episodes":10,"action_l1":0.041,
 "action_l1_per_joint":[6 floats],"chunk_l1_by_horizon":[H floats],"gripper_l1":0.07}
```
- E001은 100 ep 전체 학습이므로 `split:"train"`으로 기록해야 함

### 2.5 `rollouts_sim.jsonl` / `rollouts_real.jsonl` — rollout 1회 = 1줄
```json
{"ckpt_step":70000,"model":"A","condition":"nominal","position_id":4,"trial":1,"seed":1004,
 "success":true,"n_objects_placed":3,"episode_len_steps":612,"duration_s":20.4,
 "failure_tag":null,"video":"W07_A_nominal_p4_t1.mp4","notes":""}
```
- [rollout_schema.py](../src/aiku_hands/eval/rollout_schema.py)로 검증함. 성공률·CI는 저장하지 않고 매번 다시 계산함

## 3. lerobot stdout 파서 / Parser ([lerobot_log_parser.py](../src/aiku_hands/logging/lerobot_log_parser.py))
- 📏 로그 형식 (0.7.0):
```
INFO 2026-11-27 01:45:31 ot_train.py:894 step:20 smpl:160 ep:0 epch:0.00 loss:25.027 grdn:444.021 lr:1.0e-05 data_s:0.149 prep_s:0.019 updt_s:0.533 step_s:0.701 smp/s:11 l1_loss:0.827 kld_loss:2.420
```
- `step:` 토큰 위치를 기준으로 `(\S+?):(\S+)` 정규식으로 파싱함. tqdm 출력이 같은 줄 앞에 붙어 나와도 처리됨
- alias: `grdn→grad_norm`, `updt_s→update_s`, `smp/s→samples_per_s`, `epch→epoch`, `ep→episodes_seen`, `smpl→samples_reported`
- `\r`(tqdm)과 `\n`을 모두 줄 구분자로 씀
- 무시하는 줄: `objc[...]`, `Redirects are currently not supported`, 순수 tqdm 프레임
- checkpoint 문구 (📏 `lerobot/scripts/lerobot_train.py:936`): `Checkpoint policy after step {step}` → 경로는 `output_dir/checkpoints/{step:06d}` (`get_step_identifier`, 자릿수 = max(6, len(str(total_steps))))
- held-out loss 문구 (📏 `:928`): `step {step}: eval_loss={x:.4f}`
- 종료 문구: `End of training`

## 4. Step 복원 / Step reconstruction (중요)
- 📏 `MetricsTracker.__str__`가 `step`, `smpl`, `ep`을 `format_big_number(precision=0)`로 출력함 → 1000 이상은 `14K`처럼 표시되어 정확한 값을 알 수 없음
- 또한 f-string `.0f`는 banker's rounding을 씀 → 1500, 2000, 2500이 모두 `2K`로 출력됨
- 복원 규칙 ([ingest.py](../src/aiku_hands/logging/ingest.py)):
  1. 1000 미만은 그대로 씀 (정확)
  2. 그 외에는 후보 $s = s_{\text{prev}} + k\cdot\text{log\_freq}$ ($k=1..10$) 중 `format_big_number(s) == 출력값`인 첫 후보를 씀. 후보 순서는 타이밍 힌트 $k_{\text{hint}} = \Delta t_{\log} / (\text{log\_freq}\cdot \overline{\text{step\_s}})$에 가까운 순서로 정렬함 (줄이 빠졌을 때 대비)
  3. 맞는 후보가 없으면 가장 가까운 log_freq 배수를 쓰고 `step_exact=false`로 표시함
  4. `Checkpoint policy after step N`(정확한 값)이 나오면 그 값으로 다시 고정함
- 테스트: [tests/test_ingest.py](../tests/test_ingest.py)

## 5. 구현 단계 / Stages
- **v0 (현재)** [scripts/run_train.py](../scripts/run_train.py)
  - preflight (다른 lerobot-train 실행 중 / 디스크 < 5 GB / output_dir 존재 / train.jsonl 비어 있지 않음 → 거부)
  - manifest·config 기록 → subprocess 실행 → stdout tee (`logs/<run_id>.log`) → 실시간 ingest → sysmon 스레드
  - 종료 시 `run_end`, registry status 갱신 (`done`/`failed`/`aborted`)
  - `--dry-run`: 학습 없이 manifest 생성과 parser(실측 fixture)만 검증함
- **v1 (선택)**: in-process hook. 매 step loss, layer group별 grad·weight norm, MPS 메모리 기록. v0과 같은 seed로 돌려 loss가 일치함을 확인한 뒤 도입함
- **backfill**: [scripts/backfill_from_log.py](../scripts/backfill_from_log.py) — `ts`/`wall_s`는 로그 헤더 시각 기준임 (`wall_s`는 로그의 첫 INFO 줄 기준이며, 실제 프로세스 시작 시각보다 늦을 수 있음)

## 6. manifest.json
- `git_commit`, `git_dirty`, `git_branch`, `cmdline`(절대경로로 해석된 값), `wrapper_cmdline`, `seed`
- `python`, `torch`, `lerobot`, `mps_available`, `pip_freeze` (→ `experiments/<ID>/env/pip_freeze.txt`) + sha256
- `host`: hw_model, cpu_brand, mem_gb, os, macos
- `dataset`: repo_id, root, `meta/info.json`·`stats.json` sha256, n_episodes, n_frames, fps
