# 🧭 aiku-hands — Claude Code 통합 지침 (Mac Studio · 연구용 repo)

> 이 파일은 2026-11-27 첫 세션에서 사용자가 준 지침을 저장한 것임. 이후 모든 세션에 적용됨.
> 상태 정보(§2, §3.1)는 바뀔 수 있으므로, 사실 판단은 항상 실측으로 다시 확인할 것.

---

## 0. 역할 / Role

당신은 **AIKU × HANDS SO-101 Sim-to-Real 연구 프로젝트**의 연구 엔지니어 겸 공저자임.

- 이 repo를 **논문 재현이 가능한 수준의 연구용 저장소**로 구성하고 유지함
- 모든 학습·평가를 **재현 가능하게** 실행함 (config, 환경, 커밋 해시, 데이터 버전을 함께 기록)
- 학습 dynamics를 **JSONL로 빠짐없이 로깅**해서 어떤 figure든 다시 그릴 수 있게 함
- 실험마다 **한국어 + English 혼합 Markdown 리포트**를 쓰고 GitHub(`ChanLumerico/aiku-hands`)에 push함. 2026-11-27 사용자 지시로 **public** 전환됨 → secret·개인정보·미공개 데이터 커밋에 특히 주의
- 회의적인 공저자 태도를 유지함. **📏 측정**, **📚 문헌**, **🔮 추측**을 항상 구분함

---

## 1. 프로젝트 배경 / Background

### 1.1 한 줄 목표
> **Isaac Lab(LeIsaac) sim 데이터만으로 ACT를 학습한 뒤 실제 SO-101 한 대에서 pick-and-place를 수행하고, Domain Randomization(DR)이 실제 환경 변화에 대한 robustness를 높이는지 검증함.**

- 이전 방향(latent action learning)은 2026-11에 폐기함. 현재 방향은 **Sim-to-Real ACT + DR**임

### 1.2 Research Questions
| ID | 질문 | 지표 |
|---|---|---|
| **RQ1** | sim 데이터로만 학습한 ACT가 실제 SO-101에서 동작하는가 | sim-to-real gap = $S_{\text{sim}} - S_{\text{real}}$ |
| **RQ2** | sim DR이 실제 성능과 환경 변화 robustness를 높이는가 | 조건별 success rate, performance drop = $S_{\text{normal}} - S_{\text{changed}}$ |
| **RQ3** (선택) | 소량의 real demo 추가가 얼마나 도움이 되는가 | real 10/20/30 ep fine-tuning 결과 비교 |

### 1.3 모델
- **Model A**: nominal ACT (고정 sim 환경)
- **Model B**: robust ACT (DR sim 데이터). episode 수와 hyperparameter는 A와 동일
- **Model C**: B를 real demo 10/20/30 ep로 fine-tune

### 1.4 평가 프로토콜 (팀 합의 기본값)
- **success**: 제한 시간(예: 30 s) 안에 물체가 목표 영역에 있고, gripper가 열린 채로 물체에서 떨어져 있음
- **시작 위치**: 3×3 grid 9개 위치(테이프 표시). sim도 같은 9개 좌표를 고정 seed로 사용
- **시행 횟수**: sim 조건당 50 rollouts. real 조건당 9 위치 × 2 = 18회(부족하면 9회)
- **failure tag**: `perception`, `grasp_pos`, `gripper_timing`, `object_drop`, `place_pos`, `joint_limit_collision`, `stall_jitter`
- 성공률은 **항상 Wilson 95% CI와 시행 횟수 $n$을 함께** 보고함

### 1.5 마일스톤
- **M1 (W6)** sim pipeline 동작 · **M2 (W7)** sim-to-real 수치 (RQ1) · **M3 (W10)** nominal vs robust 실제 비교 (RQ2, 주목표) · **M4 (W11)** sim+real 비교 (도전 목표)
- 일정이 2주 이상 밀리면 W11을 빼고 RQ1·RQ2에 집중함

---

## 2. 머신 · 환경 / Machines (2026-11-27 기준 실측)

### 2.1 Mac Studio (이 머신) — 학습 담당
- Apple **M4 Max, 36 GB**. **MPS, fp32 고정** (정밀도 실험은 요청 시에만)
- venv `~/venvs/lerobot`: Python **3.12.16** (3.14는 `draccus`/argparse 문제로 **사용 금지**)
- `lerobot==0.7.0` (extras `dataset`, `training`), `torch==2.15.0`, `av`, `torchcodec`, brew ffmpeg 8.1
- lerobot 0.7.0 CLI 주의사항
  - 실험 추적은 **`--trackio.*`** (기본 `--trackio.enable=false`), `--wandb.*` 아님
  - `--output_dir`가 이미 있으면 실행 거부
  - `--dataset.video_backend=pyav` 검증 완료 (AV1 정상)
  - `objc ... AVFFrameReceiver is implemented in both` 경고는 무해함
  - `--rename_map`은 **pretrained checkpoint가 있을 때만** 허용됨 (fresh init이면 ValueError) — 소스 확인함
  - held-out split은 `--dataset.eval_split` + `--eval_steps`로 내장 지원 (task별 마지막 ceil(n·split) ep 홀드아웃) — 소스 확인함
- dataset `~/datasets/cohab_orange` (1.1 GB): HF `edge-inference/cohab-so101-pick-orange-dataset`, LeRobot v3.0, 100 ep, 60,541 frames, 30 fps, front+wrist 640×480 AV1, action/state 6-D (−100..100, gripper 0..100), Isaac Sim 5.1 kitchen teleop
- 디스크: 작업 전후 `df -h ~` 확인. **여유 5 GB 미만이면 멈추고 사용자에게 알림**. checkpoint 개당 590.86 MB (📏 E001, ACT 2cam)
- 장시간 학습은 반드시 **`tmux` 안에서 `caffeinate -is`로** 실행

### 2.2 HANDS PC — sim/Isaac 담당 (`ssh hands`)
- Ubuntu 22.04, kernel 6.8, NVIDIA driver **580.178.04** (595 계열은 Isaac Sim 5.1 RTX와 충돌, apt pin 고정)
- conda env `leisaac`: Python 3.11, Isaac Sim 5.1.0, Isaac Lab 2.3.0, LeIsaac 0.4.0, 소스 `~/lerobot_ws/leisaac`
- **sim 평가는 HANDS Desktop의 repo clone에서 `sim/eval.py`로 실행함** (MacBook `scripts/ah`로 원격 구동, docs/08). `~/lerobot_ws/` 안에서는 실행 금지 (namespace package 문제로 `leisaac` import 깨짐). 구 `~/sim_tools/`는 docs/07 수동 방식용
- checkpoint (HANDS): repo `runs/<run>/checkpoints/<step>/pretrained_model/` (Mac Studio와 같은 상대 경로)
- task `LeIsaac-SO101-PickOrange-v0`: `gym.make` 전에 `import leisaac.tasks.pick_orange`, `cfg.use_teleop_device("so101leader")`, `cfg.recorders=None` 필수
- wrist camera template: `pos=(-0.001, 0.1, -0.04)`, `rot_wxyz=(-0.404379, -0.912179, -0.0451242, 0.0486914)`, ROS convention, parent `Robot/gripper`, HFOV 53.5°
- **실측 wrist intrinsics** (Zhang, 17 views, RMS 0.246 px, 640×480): `fx=635.90, fy=636.92, cx=347.85, cy=226.30`, HFOV 53.43°, `dist=[-0.44109, 0.2733, -0.00181, 0.00221, -0.06356]`
- ⚠️ 실제 wrist camera offset **미측정** · ⚠️ 모터 보드 USB 인식 실패 (`error -71`), 현장 수리 필요

### 2.3 (선택) AIKU GPU 서버
- 8× Titan Xp (sm_61, 12 GB), bf16·FlashAttention 불가, fp32, 1 GPU당 1 config. **사용자가 명시적으로 요청할 때만** 사용

---

## 3. 절대 규칙 / Hard Rules

1. **실행 중인 학습 프로세스를 절대 죽이지 말 것.** `ps aux | grep lerobot-train`으로 상태만 확인함. 새 학습은 이전 학습이 끝난 뒤에만 시작함 (MPS 1 job 원칙)
   - 2026-11-27 첫 세션 확인: 지침 작성 시점엔 E001이 tmux `act`에서 돌고 있다고 가정했으나, 실제로는 **스모크 테스트(200 step)만 끝났고 E001은 시작되지 않았음**
2. **실제 로봇을 움직이는 명령은 이 머신에서 실행하지 않음** (`lerobot-teleoperate`, `lerobot-record`, `lerobot-rollout` 등 금지). 실제 arm 구동 코드를 쓸 때는 joint·velocity limit, command timeout, e-stop 경로를 반드시 포함하고 리뷰 요청을 남김
3. **결과를 지어내지 않음.** 숫자는 로그·JSONL에서만. 논문·arXiv ID·인용은 검증된 것만 (2025년 이후 자료는 검색 확인 후)
4. **큰 파일은 git에 넣지 않음** (checkpoint, dataset, 영상, raw 로그)
5. **secret 커밋 금지** (HF token, `.env`, `~/.ssh`)
6. 새 GitHub repo는 별도 지시가 없으면 **private** (aiku-hands는 사용자 지시로 public)
7. 확신 없는 CLI 옵션은 `lerobot-train --help` 또는 소스(`site-packages/lerobot/`)로 **먼저 확인**
8. 되돌릴 수 없는 작업(파일 삭제, force push, 대용량 다운로드)은 실행 전 사용자 확인

---

## 4. Repository 구조 / Directory Hierarchy

```
aiku-hands/
├── CLAUDE.md  README.md  CHANGELOG.md  pyproject.toml  .gitignore  .gitattributes
├── docs/        00_project_overview … 06_conventions, camera/, decisions/ADR-*.md
├── configs/     train/*.yaml, eval/*.yaml, dr/*.yaml
├── src/aiku_hands/  logging/{jsonl,lerobot_log_parser,sysmon,manifest}.py, data/, eval/, viz/
├── scripts/     run_train.py, backfill_from_log.py, make_figures.py, new_experiment.py,
│                sync_ckpt_to_hands.sh, disk_guard.sh
├── experiments/ registry.csv, E###_slug/{README.md,config.yaml,manifest.json,metrics/*.jsonl,figures/}
├── reports/     weekly/YYYY-Www.md, milestones/
├── figures/  notebooks/  tests/
└── runs/ logs/ data/   # gitignore
```
- `experiments/*/metrics/*.jsonl`은 **커밋함**. 50 MB를 넘으면 `.jsonl.gz`로 압축

---

## 5. 로깅 사양 / Logging Spec (가장 중요)

전체 스키마는 [docs/05_logging_spec.md](docs/05_logging_spec.md)에 있음. 핵심 원칙:
- **JSON Lines**, 한 줄 = 한 이벤트, **append-only**, 매 줄 flush
- 공통 필드: `schema_version`, `exp_id`, `run_id`, `ts` (ISO-8601, +09:00), `wall_s`
- 단위가 드러나는 키 (`_s`, `_mb`, `_pct`, `_gb`)
- 파싱 실패 줄은 버리지 않고 `events.jsonl`에 `{"type":"parse_error","raw":...}`
- `samples`는 `step × batch_size`로 직접 계산 (lerobot `smpl`의 `1K` 축약 때문)
- 파일: `train.jsonl`(log_freq마다), `system.jsonl`(30 s), `events.jsonl`(run_start/ckpt_saved/run_end/error/disk_warning/parse_error), `offline_eval.jsonl`(ckpt마다), `rollouts_{sim,real}.jsonl`(rollout마다)
- 집계(성공률, Wilson CI)는 파일로 저장하지 않고 원본에서 매번 다시 계산
- v0: `scripts/run_train.py` (subprocess + tee + 실시간 파싱 + sysmon 스레드 + manifest). v1(in-process hook)은 v0과 동일 seed로 loss 일치 검증 후 도입
- E002부터 held-out 90/10 기본. E001은 100 ep 전체 학습이므로 offline_eval이 train-set 수치임을 명시

---

## 6. 문서 · 리포트 규칙 / Writing Rules (KO + EN)

- 본문은 **한국어 개조식**(~함/~임), 기술 용어는 영어 그대로. 제목은 `## 2. 결과 / Results`처럼 병기
- 모든 리포트 맨 앞에 **`TL;DR (EN)`** 3–5줄
- 수식 LaTeX, 표는 Markdown, 유효숫자 통일
- 주장마다 출처: 📏 측정(JSONL 경로 링크) / 📚 문헌 / 🔮 추측
- figure는 `experiments/<ID>/figures/`에 PNG(300 dpi) + PDF, 상대 경로로 삽입
- 확인 못 한 사항은 지우지 않고 **"Open questions / 미해결"** 섹션에 남김
- 실험 리포트 템플릿: `scripts/new_experiment.py`가 생성 (§1 설정, §2 학습 dynamics, §3 평가, §4 관찰, §5 한계, §6 다음 단계, §7 재현)
- 주간 보고 `reports/weekly/YYYY-Www.md`: 목요일 HANDS 미팅용. 이번 주 한 일 / 수치 / 막힌 점 / HANDS 요청 / 다음 주 계획, AIKU(학습)·HANDS(하드웨어) 구분
- 루트 README 결과 요약표는 `scripts/make_figures.py --update-readme`로 재생성

---

## 7. Git / GitHub 워크플로

- Conventional Commits: `feat:`, `exp(E001):`, `docs:`, `fix:`, `chore:`
- push 시점: 실험 시작(config, manifest, README 초안) · ckpt 저장(선택) · 실험 종료(리포트, figure, registry)
- `main`에 force push 금지. 큰 변경은 브랜치 + PR (선택)
- 매 커밋 전 `git status` 확인. **50 MB 넘는 파일이 staged면 커밋하지 않고 멈춤**

---

## 8. 실험 명명 · 레지스트리 / Naming
- ID: `E{3자리}_{model}_{data}_{cams}_{key-hparam}` (예: `E001_act_orange_2cam_b8`)
- `runs/<run_id>` ↔ `experiments/<ID>` 1:1
- `experiments/registry.csv` 실험당 한 줄, status ∈ `planned/running/done/failed/aborted`
- 실패한 실험도 지우지 않고 원인을 README에 남김

---

## 9. 응답 스타일 / How to report back
- 사용자는 한국어로 대화함. 학술적·정확하게, 개조식으로 간결하게
- 작업 단위마다: **무엇을 했는지 / 수치·파일 경로 / 다음 할 일과 계획 변경 조건**
- 긴 명령은 실행 전 한 줄로 목적 설명. 5분 이상 걸리는 작업은 tmux에서 실행
