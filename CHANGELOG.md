# CHANGELOG
날짜별 주요 변경과 결정 로그 (최신이 위).

## 2026-11-27
- **repo bootstrap**: CLAUDE.md, 디렉터리 구조, `aiku_hands` 패키지, 문서 00–06, camera, ADR-0001~0003
- **로깅 v0**: `run_train.py` (preflight + manifest + live JSONL + sysmon), `backfill_from_log.py`, `make_figures.py`, 28 unit tests
- **발견 (📏 소스)**: lerobot 0.7.0은 `step`/`smpl`/`ep`를 `format_big_number`로 축약 출력함 (banker's rounding 포함) → step 복원 로직과 `step_exact` 플래그를 추가함
- **발견 (📏 소스)**: `--dataset.eval_split` + `--eval_steps`로 held-out loss를 내장 지원함. `--rename_map`은 fresh init에서 쓸 수 없음
- **E000** (smoke, 200 step): tmux 기록에서 backfill. 2.94 step/s, compute-bound, loss 25.0→3.55
- **상태 정정**: 지침은 E001이 01:50부터 실행 중이라고 가정했으나, 실제로는 미실행이었음 → E001을 `planned`로 등록함
- **데이터**: cohab orange의 episode 길이 long-tail을 확인함 (ep33 2564 frames, ep40 1912 frames)
