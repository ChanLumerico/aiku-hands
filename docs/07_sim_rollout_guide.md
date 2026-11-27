# 07. E001 정책을 LeIsaac sim에서 롤아웃하기 / Running E001 in the LeIsaac simulator
> ➡️ **원격·반복 평가는 [docs/08](08_sim_eval.md)의 `scripts/ah eval`을 씀.** 이 문서는 GUI 시연용 수동 2-터미널 방식과 배경 설명임.
> 2026-11-27 HANDS GUI pilot에서 발견: sim reset 자세(0 rad)가 dataset 시작 자세와 달라 OOD → `--warmup_steps 120` 필요, reset 직후 첫 프레임은 텍스처 없음.
> TL;DR (EN): Use checkpoint `070000` (lowest train loss; there is no held-out split, so "best" = last).
> LeIsaac's built-in `--policy_type=lerobot-act` path does **not** work with our lerobot 0.7.0 checkpoint
> (it speaks lerobot v0.3.3's gRPC protocol), so we run a 2-process bridge on the HANDS PC: a policy server in
> a Python 3.12 / lerobot 0.7.0 env, and a LeIsaac rollout script in the `leisaac` env. The server was tested
> on the Mac Studio with the real checkpoint; the sim script has not yet been run on the HANDS PC.

- 대상: HANDS PC에서 직접 롤아웃할 팀원 (AIKU·HANDS)
- 소요 예상: 처음 설치 30–60분 (🔮), 이후 롤아웃 1회는 수십 초

---

## 0. 한눈에 보기 / At a glance
```
 Mac Studio                                HANDS PC (Ubuntu, RTX)
 runs/act_orange_2cam_b8/                  ~/policies/E001/070000/pretrained_model/   (197 MB)
   checkpoints/070000/pretrained_model ──▶        │
                (복사)                            ▼
                                  [터미널 A] conda env lerobot070 (Python 3.12, lerobot 0.7.0)
                                     python ~/sim_tools/act_policy_server.py --device cuda
                                                  ▲ TCP 127.0.0.1:5765 (obs → 100×6 action chunk)
                                                  ▼
                                  [터미널 B] conda env leisaac (Python 3.11, Isaac Sim 5.1)
                                     cd ~/sim_tools && python leisaac_act_rollout.py --enable_cameras
```

## 1. 어떤 checkpoint를 쓰나 / Which checkpoint
| step | train L1 (📏) | 경로 (Mac Studio, repo 기준) |
|---|---|---|
| **070000 (권장, = `last`)** | **0.140** | `runs/act_orange_2cam_b8/checkpoints/070000/pretrained_model/` |
| 056000 | 0.154 | `runs/act_orange_2cam_b8/checkpoints/056000/pretrained_model/` |
| 042000 | 0.175 | `runs/act_orange_2cam_b8/checkpoints/042000/pretrained_model/` |
| 028000 | 0.208 | `runs/act_orange_2cam_b8/checkpoints/028000/pretrained_model/` |
| 014000 | 0.280 | `runs/act_orange_2cam_b8/checkpoints/014000/pretrained_model/` |

- ⚠️ E001에는 held-out split이 없음 → "best"는 **train loss 기준**임. 일반화 기준 best는 sim 성공률로 정해야 함
  - 70k가 잘 안 되면 42k·56k도 같은 방식으로 돌려 비교함
- 전송할 것은 `pretrained_model/` 폴더(197 MB)뿐임. `training_state/`(394 MB)는 학습 재개용이라 필요 없음
- 폴더 구성: `config.json`, `model.safetensors`, `policy_preprocessor*.json/.safetensors`(정규화 통계), `policy_postprocessor*`, `train_config.json`
- checkpoint는 git에 없음 (용량 때문에 제외). Mac Studio 로컬에만 있음

## 2. 왜 LeIsaac 기본 평가 스크립트를 그대로 못 쓰나 / Why not LeIsaac's built-in path
| 항목 | LeIsaac v0.4.0 `policy_inference.py --policy_type=lerobot-act` | 우리 checkpoint (lerobot 0.7.0) |
|---|---|---|
| 통신 | lerobot **v0.3.3** async-inference gRPC + pickle (`lerobot.scripts.server.helpers`) | 0.7.0 policy server는 Zenoh 기반. 해당 gRPC 서비스·모듈 없음 |
| 정규화 통계 | v0.3.3은 `model.safetensors` 안에 들어 있다고 가정 (🔮) | 별도 `policy_preprocessor_*.safetensors`에 있음. 모델 가중치에는 없음 (📏 확인) |
| Python | leisaac env 3.11 | lerobot 0.7.0은 **Python ≥3.12** 필요 |
- 📚 근거: LeIsaac v0.4.0 소스 `source/leisaac/leisaac/policy/service_policy_clients.py` ("Target Commit: lerobot v0.3.3"), `policy/lerobot/__init__.py`, lerobot 0.7.0 `scripts/lerobot_policy_server.py` (ZenohTransport)
- 그래서 **정책 서버는 별도 env에서 돌리고, sim은 아주 작은 TCP 프로토콜로 통신함** ([sim/act_bridge.py](../sim/act_bridge.py), stdlib + numpy만 씀)

## 3. 준비 / Setup (HANDS PC, 최초 1회)

### 3.1 checkpoint 복사
- 현재 Mac Studio → `ssh hands`는 **publickey 인증 실패**임 (2026-11-27 확인). 둘 중 하나로 복사함
  - (a) 아무 경로로 `pretrained_model/` 폴더를 HANDS의 `~/policies/E001/070000/pretrained_model/`에 둠 (MacBook 경유, USB 등)
  - (b) Mac Studio 공개키를 HANDS `~/.ssh/authorized_keys`에 등록한 뒤, Mac Studio에서:
    ```bash
    bash scripts/sync_ckpt_to_hands.sh E001 runs/act_orange_2cam_b8 070000        # dry-run
    bash scripts/sync_ckpt_to_hands.sh E001 runs/act_orange_2cam_b8 070000 --go   # 실제 전송
    ```
- 복사 후 확인: `ls ~/policies/E001/070000/pretrained_model` → `config.json`, `model.safetensors` 등 7개 파일

### 3.2 정책 서버용 env (Python 3.12 + lerobot 0.7.0)
```bash
conda create -n lerobot070 python=3.12 -y
conda activate lerobot070
pip install 'lerobot==0.7.0'
python -c "import torch, lerobot; print(torch.__version__, torch.cuda.is_available(), lerobot.__version__)"
# 기대: <torch 버전> True 0.7.0
```
- **leisaac env에는 설치하지 말 것** (Python 3.11이고, Isaac 의존성이 깨질 수 있음)
- `torch.cuda.is_available()`이 False이면 CUDA wheel을 다시 설치함. 드라이버 580은 최신 CUDA wheel을 지원할 것으로 봄 🔮

### 3.3 스크립트 배치
```bash
cd ~ && git clone https://github.com/ChanLumerico/aiku-hands.git   # public repo
mkdir -p ~/sim_tools
cp ~/aiku-hands/sim/act_bridge.py ~/aiku-hands/sim/act_policy_server.py ~/aiku-hands/sim/leisaac_act_rollout.py ~/sim_tools/
```
- 반드시 `~/sim_tools/`에서 실행함. `~/lerobot_ws/` 안에서 실행하면 `import leisaac`가 깨짐 (docs/02)

## 4. 실행 / Run

### 터미널 A — 정책 서버
```bash
conda activate lerobot070
python ~/sim_tools/act_policy_server.py --ckpt ~/policies/E001/070000/pretrained_model --device cuda
```
- `listening on 127.0.0.1:5765`가 나오면 준비 완료
- checkpoint에는 `device: mps`가 저장되어 있지만, 서버가 `--device`로 덮어씀

### 터미널 B — LeIsaac 롤아웃 (GUI, 미팅 시연용)
```bash
conda activate leisaac
cd ~/sim_tools
python leisaac_act_rollout.py \
  --task LeIsaac-SO101-PickOrange-v0 \
  --enable_cameras --device cuda \
  --eval_rounds 10 --seed 1000 \
  --action_horizon 100 --step_hz 60 \
  --exp_id E001 --ckpt_step 70000 \
  --out rollouts_sim_pilot.jsonl
```
- `--enable_cameras`: Isaac Lab에서 카메라 센서를 쓰려면 필수임 (정책 입력이 front·wrist 이미지)
- `--headless`를 붙이면 GUI 없이 돎. 많이 돌릴 때는 `--step_hz 0`으로 실시간 대기를 없앰
- 터미널 A에 `client connected` 로그가 뜨고, 에피소드가 끝날 때 `client disconnected (… mean … ms)`가 나옴
- 에피소드마다 `[rollout] ep 3: success=True steps=612 -> 2/3` 형식으로 출력됨

### 시도해 볼 옵션
| 옵션 | 의미 | 언제 |
|---|---|---|
| `--action_horizon 100` (기본) | chunk 100개를 다 실행한 뒤 다시 추론 (학습 설정과 같음) | 먼저 이걸로 시작 |
| `--action_horizon 50` / `25` | 더 자주 다시 추론 (closed-loop에 가까움) | 70k가 중간에 헤매거나 놓칠 때 |
| `--ckpt` 042000 / 056000 | 이전 checkpoint | 70k 결과가 나쁠 때 |

## 5. 결과 / Results
- `rollouts_sim_pilot.jsonl`에 에피소드마다 1줄 (스키마는 docs/05 §2.5와 같고, 추가 필드가 있음)
- **success 정의 (LeIsaac `task_done`, 📚 소스 확인)**: 오렌지 3개가 **모두** 접시 중심 기준 x·y ±10 cm, z ±7 cm 안에 있고, **로봇이 rest pose로 돌아옴**
  - 이 조건으로 에피소드가 종료되면 success, `--episode_length_s`(기본 60 s)를 넘기면 timeout(실패)
- 결과 파일은 Mac Studio repo의 `experiments/E001_act_orange_2cam_b8/metrics/`에 넣어 주면 Wilson CI로 정리함
  - 예: 10회 중 3회 성공이면 `0.300 [0.108, 0.603] (n=10)` — n이 작으면 CI가 매우 넓음

## 6. 문제 해결 / Troubleshooting
| 증상 | 확인할 것 |
|---|---|
| `ConnectionRefusedError` | 터미널 A 서버가 떠 있는지, 포트(5765)가 같은지 |
| `AuthenticationError` | 두 터미널의 `AIKU_BRIDGE_KEY` 환경변수가 같은지 (기본값이면 둘 다 설정 안 함) |
| `missing camera 'front'` | 카메라 이름이 `front`/`wrist`인지 (LeIsaac single-arm 템플릿 기본값) |
| `expected HWC (480, 640, 3)` | 카메라 해상도가 640×480인지 (템플릿 기본값) |
| `ModuleNotFoundError: lerobot` (터미널 B) | 정상임. sim 쪽은 lerobot이 필요 없음. 서버만 lerobot070 env에서 돌림 |
| 팔이 엉뚱한 방향으로 크게 움직임 | 단위 변환 문제일 수 있음. 스크립트는 LeIsaac의 `convert_leisaac_action_to_lerobot`/`convert_lerobot_action_to_leisaac`를 그대로 씀. 결과를 공유해 주면 같이 확인함 |
| 첫 추론이 느림 | 정상 (warm-up). Mac MPS에서 첫 호출 약 1.2 s, 이후 약 27 ms (📏) |

## 7. 검증 상태 · 주의 / Verification status & caveats
- 📏 **검증함 (Mac Studio, MPS)**: 서버가 실제 70k checkpoint를 로드하고, dataset ep0의 실제 프레임(t=0/150/300)에 대해 (100, 6) action chunk를 돌려줌
  - GT action 대비 joint별 평균 오차 1.2–8.2 (lerobot 단위, −100..100 스케일). train-set 프레임이므로 낙관적인 수치임
- 🔮 **미검증**: `leisaac_act_rollout.py`는 HANDS PC에서 아직 한 번도 돌려 보지 않음
  - LeIsaac v0.4.0 `policy_inference.py`의 env 구성, 관측 키, 단위 변환, success/timeout 로직을 따라 썼지만 첫 실행에서 오류가 날 수 있음
- 이번 롤아웃은 **pilot**임. 프로토콜(docs/04)의 9-grid 고정 좌표가 아니라 LeIsaac 기본 reset randomization을 씀 → `position_id: null`, `condition: nominal_pilot`
- E001은 **front + wrist 2-cam** 정책임. 실제 로봇 이식(RQ1)은 front 카메라 위치를 sim과 맞출 수 있을 때만 의미가 있음
- 🔮 sim step 1회 = dataset 1 frame으로 가정함 (LeIsaac 변환 스크립트가 env step마다 frame을 기록함). dataset의 "30 fps" 표기와 실제 sim dt의 관계는 확인 필요

## Open questions / 미해결
- Mac Studio ↔ HANDS PC SSH 키 등록 (checkpoint 전송 자동화)
- 9-grid 좌표를 LeIsaac reset event에 넣는 방법 (protocol 준수 rollout용)
- 실패 에피소드의 failure tag를 자동으로 달지, 영상을 보고 수동으로 달지
