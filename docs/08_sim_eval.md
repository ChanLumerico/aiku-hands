# 08. 통합 sim 평가 (원격, 진행 막대) / One-command sim evaluation from the MacBook
> TL;DR (EN): `scripts/ah` (run on the MacBook) drives `sim/eval.py` on the HANDS PC over ssh: it syncs
> checkpoints from the Mac Studio, starts a sweep in tmux (policy server + headless LeIsaac per condition),
> shows live progress bars, and fetches JSONL/videos back. Rollouts run headless from the repo clone on the
> HANDS Desktop. Aggregates (success, Wilson CI) are always recomputed from `rollouts/*.jsonl`.
> Pipeline tested end-to-end with mock/stub Isaac (📏 2026-11-27); first real Isaac run pending.

- 대상: 원격(ssh)으로만 HANDS PC에 접속하는 상황. **실기체는 구동하지 않음**
- 수동 2-터미널 방식(docs/07)을 대체함. 내부 구성은 docs/07과 같음 (`act_policy_server.py` ↔ TCP ↔ `leisaac_act_rollout.py`)

---

## 0. 구성 / Layout
```
 MacBook ── scripts/ah ──ssh──▶ HANDS PC: <Desktop>/aiku-hands   (xdg-user-dir DESKTOP; ~/Desktop 또는 ~/바탕화면)
   │                               tmux "ah-eval": python3 sim/eval.py run <config>
   │                                 ├─ conda run -n lerobot070  sim/act_policy_server.py   (ckpt마다 1회)
   │                                 └─ conda run -n leisaac     sim/leisaac_act_rollout.py --headless (조건마다 1회)
   └──ssh──▶ Mac Studio: runs/<run>/checkpoints/<step>/pretrained_model  ──(ah sync, MacBook 경유)──▶ HANDS repo 같은 경로
```
- `sim/eval.py`, `scripts/ah`는 **stdlib만** 씀 → HANDS 시스템 `python3`(3.10), MacBook `python3`에서 바로 돎
- checkpoint 위치 (HANDS): repo 안 `runs/<run>/checkpoints/<step:06d>/pretrained_model/` (Mac Studio와 같은 상대 경로, gitignore)
- repo가 `~/lerobot_ws/` 밖에 있으므로 `import leisaac` namespace 문제 없음 (docs/02)

## 1. 최초 1회 / One-time setup (MacBook)
```bash
git clone https://github.com/ChanLumerico/aiku-hands.git && cd aiku-hands   # MacBook에도 clone
scripts/ah setup          # HANDS Desktop에 clone(또는 pull) + conda env·tmux·ffmpeg 확인
scripts/ah sync act_orange_2cam_b8 28000 42000 56000 70000   # 각 ~197 MB, 이미 있으면 건너뜀
```
- 기존 `~/policies/E001/070000`은 그대로 두어도 됨 (새 경로로 다시 sync하거나 `cp -a`)

## 2. 실행 / Run
```bash
scripts/ah eval configs/eval/E001_smoke.json       # 1 ep: warmup state0·텍스처·영상 확인 (먼저!)
scripts/ah eval configs/eval/E001_ckpt_sweep.json  # 4 ckpt × h100 + 70k × h50/h25, 각 20 ep
```
- `ah eval`은 tmux `ah-eval`에서 시작한 뒤 곧바로 `watch` 화면을 띄움. **Ctrl-C는 화면만 닫고 실행은 계속됨**
- 다시 보기 `ah watch` · tmux 직접 보기 `ah attach` · 중단 `ah stop` · 이어서 실행 `ah resume`
- 최신 sim 로그 `ah log` · GPU `ah gpu` · 목록 `ah list`

화면 예시 (📏 mock 실행):
```
aiku-hands sim eval · E001 · 20261127-203329_E001_ckpt_sweep
conds 2/6 · elapsed 0:41:07 · ETA 2:10:12 · GPU 87% 7.9/12.0 GB
 ✓  ck28k_h100     succ 1/20 (5%) · placed≤ 0.8   CI95 [0.01,0.24]
 ▶  ck42k_h100     ██████░░░░░░░░░░░░░░   7/20 eps · succ 0/7 (0%)
      ep   8 █████████████░░░░░░░ 2460/3480 steps · 61 st/s · infer 21 ms
 ·  ck56k_h100     pending
```

## 3. 결과 / Results
```bash
scripts/ah summary            # HANDS에서 계산한 표 (--md: Markdown 표)
scripts/ah fetch              # run 폴더(jsonl·logs·videos)를 MacBook ./runs/sim_eval/로 복사 + 표 출력
```
| 파일 (run 폴더) | 내용 |
|---|---|
| `plan.json` | 설정 전체 + git commit/dirty + host (재현용) |
| `rollouts/<cond>.jsonl` | **연구 데이터**. episode당 1줄 (docs/05 스키마 + `warmup_steps`, `state0`, `max_objects_placed`, `video`) |
| `progress.jsonl` | UI 이벤트 (연구 데이터 아님) |
| `videos/<cond>/ep###.mp4` | 정책이 실제로 본 입력 (front \| wrist), 처음 `video_eps`개 ep, 실시간 속도 (30 fps = 60 Hz / 2) |
| `dumps/<cond>/` | 추론 입력 PNG (`dump_eps`>0일 때) |
| `logs/{server,sim}_*.log` | 프로세스 전체 출력 |
- 실험 기록으로 남길 때: `rollouts/*.jsonl`을 `experiments/E001_.../metrics/rollouts_sim.jsonl`로 합쳐 Mac Studio에서 커밋 (영상은 커밋하지 않음)
- crash 후 resume하면 이전 시도는 `<cond>.crashed-N.jsonl.bak`으로 보존되고 summary에서는 빠짐

## 4. 설정 / Config (`configs/eval/*.json`)
| 키 | 기본 | 의미 |
|---|---|---|
| `conditions` | — | `[{ckpt, horizon, (episodes, seed, warmup_steps, extra_args)}]`. 같은 ckpt가 이어지면 서버 1개를 공유 |
| `episodes` / `seed` | 20 / 1000 | 조건마다 같은 seed → 같은 초기 배치 순서 (paired 비교) |
| `warmup_steps` | 120 | reset 후 dataset rest pose `[8,−99.9,96,64,15,13]` 유지 (OOD 리셋 자세·무텍스처 첫 프레임 대응) |
| `episode_length_s` | 60 | env 자체 timeout (warmup 포함). protocol 제한은 `max_policy_steps`로 따로 적용 |
| `success_def` / `success_hold_steps` / `release_dist` / `max_policy_steps` | protocol_v1 / 60 / 0.10 / 1800 | ADR-0004. 조건별 덮어쓰기 가능 |
| 조건별 `rest` / `temporal_ensemble` | — | warmup 자세 6개 값 (lerobot 단위) / ACT temporal ensembling 계수 (서버를 따로 띄우고 매 step 1 action) |
| `step_hz` | 0 | 0 = 최대 속도 (headless 평가용) |
| `video_eps` / `dump_eps` | 2 / 0 | 녹화·덤프할 앞쪽 ep 수 |
| `envs` | `lerobot070` / `leisaac` | conda env 이름 |
- `--episodes N`, `--video_eps N`으로 덮어쓸 수 있음. `--mock`: Isaac 없이 파이프라인만 시험

## 5. 지표 주의 / Metric caveats
- **success** (2026-11-28부터, [ADR-0004](decisions/ADR-0004-sim-success-protocol.md)) = `protocol_v1`: 오렌지 3개 모두 접시 box 안 + gripper·jaw가 모든 오렌지에서 0.10 m 이상 떨어짐, 1 s 연속 유지, 정책 시작 후 30 s 안
  - LeIsaac `task_done`(위 조건 + **rest 자세 ±30°**)은 `success_leisaac`로 별도 기록. 데이터 데모의 18/100만 rest box 안에서 끝나므로 주 지표로 쓰지 않음
  - 2026-11-27 sweep 1 record에는 `success_def`가 없음 → 그 `success`는 LeIsaac 기준임
  - 분석: `python3 scripts/analyze_sim_rollouts.py <run_dir | rollouts.jsonl> --ref <cond>` (Wilson CI, paired sign/McNemar)
- **`max_objects_placed`** (🔮 보조 지표): 15 step마다 접시 기준 x·y ±10 cm, z ±7 cm 안 오렌지 수의 episode 최댓값. rest 복귀 조건은 없음
  - rigid object 이름에 `orange`/`plate`가 들어간다고 가정함. 못 찾으면 `null`이고 sim 로그 `[placed] rigid objects: [...]`에 실제 이름이 찍힘
- 이번 sweep은 **pilot**임 (LeIsaac 기본 reset randomization, docs/04의 9-grid·50회 아님)
- 20 ep의 Wilson CI는 넓음 (예: 6/20 → [0.15, 0.52]). 점추정이 아니라 CI 겹침으로 비교함

## 6. 검증 상태 / Verification
- 📏 mock(`--mock`): 6 조건 sweep, crash 처리, `--resume`, `watch`(TTY·non-TTY), Ctrl-C 시 자식 프로세스 정리
- 📏 stub Isaac(가짜 `isaaclab`/`leisaac` 모듈 + 실제 `act_bridge` 서버): **실제** `leisaac_act_rollout.py`가 eval.py의 conda 명령으로 실행되어 warmup 후 `state0 = [8, −99.9, 96, 64, 15, 13]`, mp4 1280×480 30 fps, PNG dump, JSONL, placed 카운트까지 동작함
- 🔮 미검증: 실제 Isaac Sim 5.1에서의 headless 실행, `env.scene.rigid_objects` 이름, `conda run` 하의 Isaac 환경 변수

## 7. 문제 해결 / Troubleshooting
| 증상 | 원인 · 조치 |
|---|---|
| `Failed to open layer @<repo>/assets/scenes/.../scene.usd@` | LeIsaac은 `ASSETS_ROOT = <cwd의 git root>/assets`로 정함 → 다른 git repo 안에서 실행하면 경로가 틀어짐 (📏 2026-11-27). rollout이 `LEISAAC_ASSETS_ROOT`를 LeIsaac checkout의 `assets/`로 자동 고정함. 다른 위치면 환경변수로 직접 지정 |
| `policy server ... timeout` | `logs/server_ck*.log` 확인 (CUDA·lerobot env) |

## Open questions / 미해결
- `conda run`으로 띄운 Isaac이 activate 때와 같은 환경 변수를 받는지 (실패 시 `envs`에 python 절대 경로를 쓰는 옵션 추가)
- 9-grid 고정 좌표 reset event (프로토콜 rollout용)
