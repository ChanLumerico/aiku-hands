# 06. 규칙 / Conventions
> TL;DR (EN): Experiment IDs `E###_model_data_cams_hparam`, 1:1 with `runs/<run_id>`; registry.csv is the index;
> Conventional Commits; reports are Korean bullet style + English TL;DR with 📏/📚/🔮 provenance markers.

## 1. 실험 ID / Experiment IDs
- `E{3자리}_{model}_{data}_{cams}_{key-hparam}` 예: `E001_act_orange_2cam_b8`
- `run_id` = ID에서 `E###_`를 뺀 부분 (예: `act_orange_2cam_b8`)
- `runs/<run_id>`(lerobot output) ↔ `experiments/<ID>`(기록) ↔ `configs/train/<run_id>.yaml`을 1:1로 대응시킴
- 새 실험: `python scripts/new_experiment.py E002_act_orange_wrist_b8 --purpose ...`
- E000은 스모크 테스트용 번호임

## 2. registry.csv
- 열: `id,date,status,model,dataset,steps,batch,cams,best_metric,report`
- status: `planned / running / done / failed / aborted`. `run_train.py`가 running → done/failed/aborted로 자동 갱신함
- 실패한 실험도 지우지 않음. 원인을 README에 남김

## 3. Git
- Conventional Commits: `feat:`, `fix:`, `docs:`, `chore:`, `test:`, `exp(E001): ...`
- push 시점: 실험 시작(config/manifest/README 초안), checkpoint 저장(선택), 실험 종료(리포트/figure/registry)
- 커밋 전에 `git status`를 확인함. 50 MB를 넘는 staged 파일이 있으면 멈춤. `main`에는 force push하지 않음
- 커밋 대상: 코드, 문서, config, `metrics/*.jsonl`, figure (PNG/PDF), `env/pip_freeze.txt`
- 커밋 제외: `runs/`, `logs/`, `data/`, checkpoint, 영상, secret

## 4. 리포트 문체 / Writing
- 본문은 한국어 개조식(~함/~임), 기술 용어는 영어 그대로. 제목은 KO / EN 병기
- 맨 앞에 `TL;DR (EN)` 3–5줄
- 출처 표시: 📏 측정 (JSONL 경로 링크) · 📚 문헌 (검증된 것만) · 🔮 추측
- 성공률은 항상 `p [lo, hi] (n=…)` 형식으로 씀 (Wilson 95%)
- figure는 PNG(300 dpi)와 PDF를 함께 저장하고 상대 경로로 삽입함. 스타일은 `aiku_hands.viz.style`
- 확인하지 못한 사항은 "Open questions / 미해결" 섹션에 남김

## 5. 주간 보고 / Weekly report
- `reports/weekly/YYYY-Www.md` (ISO 주차), 매주 목요일 HANDS 미팅용
- 구성: 이번 주 한 일 / 수치 / 막힌 점 / HANDS에 요청할 사항 / 다음 주 계획. AIKU(학습)와 HANDS(하드웨어) 담당을 나눠서 적음
