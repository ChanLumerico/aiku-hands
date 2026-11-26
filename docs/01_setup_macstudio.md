# 01. Mac Studio 환경 / Mac Studio Setup
> TL;DR (EN): Training host. M4 Max 36 GB, MPS fp32 only. venv `~/venvs/lerobot` with Python 3.12.16,
> lerobot 0.7.0 `[dataset,training]`, torch 2.15.0. Known pitfalls listed below (all measured on 2026-11-27).

## 1. 사양 / Spec (📏 `sysctl`, manifest)
| 항목 | 값 |
|---|---|
| 모델 | Mac16,9 · Apple M4 Max · 36 GB unified |
| OS | macOS 27.3 (Darwin 27.3.0) |
| 가속 | MPS (`torch.backends.mps.is_available() == True`), **fp32 고정** |
| 디스크 | 460 GiB 중 여유 약 16–17 GB (2026-11-27, 97% 사용) |

## 2. venv
```bash
# Python 3.12 (brew python@3.12) — 3.14 금지
python3.12 -m venv ~/venvs/lerobot && source ~/venvs/lerobot/bin/activate
pip install 'lerobot[dataset,training]==0.7.0'   # training extra 없으면 accelerate ImportError
pip install -e ~/Desktop/Programming/aiku-hands'[dev]'
brew install ffmpeg   # 8.1
```
- 📏 버전: Python 3.12.16, torch 2.15.0, lerobot 0.7.0, accelerate 1.16.0, trackio 0.45.1
- 전체 freeze는 실험마다 `experiments/<ID>/env/pip_freeze.txt`에 저장됨 (`run_train.py`)

## 3. 알려진 함정 / Known pitfalls (📏 실측)
| 증상 | 원인 / 해결 |
|---|---|
| `TypeError: str \| None is not callable` | Python 3.14 + draccus/argparse. **3.12 사용** |
| `ImportError: 'accelerate' is required` | `lerobot[training]` extra 누락 → `pip install 'lerobot[dataset,training]'` |
| `unrecognized arguments: --wandb.enable=false` | 0.7.0은 `--trackio.*` 사용 (`--trackio.enable=false`) |
| `--output_dir` 존재 시 실행 거부 | 새 run마다 새 output_dir. `run_train.py` preflight가 사전 차단 |
| `objc ... AVFFrameReceiver is implemented in both` | cv2·av·brew ffmpeg dylib 중복. 무해 (parser가 무시) |
| `NOTE: Redirects are currently not supported in MacOs` | DataLoader worker마다 출력. 무해 (parser가 무시) |
| `--rename_map` + fresh init → ValueError | 0.7.0은 pretrained ckpt가 있을 때만 rename 허용 (`configs/train.py`) |

- 영상 디코딩: `--dataset.video_backend=pyav`로 AV1 정상 디코딩 확인

## 4. 장시간 학습 / Long runs
```bash
tmux new -s train
caffeinate -is python scripts/run_train.py --exp E00X --config configs/train/<run>.yaml
```
- `caffeinate -is`: idle·system sleep 방지. tmux 안에서 실행해야 터미널 종료에도 살아남음
- 실행 중 학습은 **절대 kill하지 않음**. 상태 확인은 `ps aux | grep lerobot-train`만 사용
- MPS 1 job 원칙: `run_train.py`가 다른 `lerobot-train` 프로세스를 감지하면 시작을 거부함

## 5. 처리량 기준값 / Throughput baseline (📏 [E000](../experiments/E000_act_orange_smoke/README.md))
- ACT 52M, 2 cams, bs=8, num_workers=8: steady-state `step_s` ≈ 0.340 s (≈ 2.95 step/s), `data_s` 0.003 s
- → 70k steps ≈ $70000 \times 0.340\,\text{s} \approx 6.6\,\text{h}$ (checkpoint 저장 시간 제외 🔮)
