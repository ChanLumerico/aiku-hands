# CHANGELOG
날짜별 주요 변경과 결정 로그 (최신이 위).

## 2026-11-27 (밤)
- **통합 sim 평가**: `sim/eval.py` (sweep orchestrator + 진행 막대 `watch` + Wilson `summary` + `--resume`, stdlib only) + MacBook용 `scripts/ah` (setup/sync/eval/watch/fetch) + docs/08. preset `configs/eval/E001_smoke.json`, `E001_ckpt_sweep.json`
- `leisaac_act_rollout.py`: `--warmup_steps/--rest` (HANDS GUI pilot에서 발견한 OOD reset 자세·무텍스처 첫 프레임 대응), `--progress`, `--video_dir` (ffmpeg pipe, front|wrist), `--dump_dir`, 보조 지표 `max_objects_placed` (🔮 best-effort), `state0` 기록
- 📏 mock + stub-Isaac으로 실제 rollout 스크립트 경로를 끝까지 검증함. 실제 Isaac 실행은 미검증
- HANDS 작업 위치를 `~/sim_tools/` → Desktop repo clone으로 변경함 (CLAUDE.md §2.2)

## 2026-11-27 (저녁)
- **sim 롤아웃 브리지**: `sim/act_policy_server.py` (lerobot 0.7.0) + `sim/leisaac_act_rollout.py` (leisaac env) + docs/07. LeIsaac v0.4.0 내장 lerobot 클라이언트가 lerobot v0.3.3 gRPC 전용이라 0.7.0 checkpoint와 호환되지 않음을 확인함. 서버는 Mac에서 실제 ckpt로 검증함. sim 스크립트는 HANDS PC에서 미검증

## 2026-11-27 (오후)
- **E001 완료**: 02:14→08:54, exit 0, 6.662 h. train L1 0.695→0.140 (미수렴), KL이 35.5k step 이후 <5e-4 (collapse), ckpt 5 × 590.86 MB
- repo public 전환 (사용자 지시)
- fix: manifest의 `git_dirty`를 env 파일 쓰기 전에 계산함 / viz: kld가 0으로 출력되면 symlog 축을 씀

## 2026-11-27
- **repo bootstrap**: CLAUDE.md, 디렉터리 구조, `aiku_hands` 패키지, 문서 00–06, camera, ADR-0001~0003
- **로깅 v0**: `run_train.py` (preflight + manifest + live JSONL + sysmon), `backfill_from_log.py`, `make_figures.py`, 28 unit tests
- **발견 (📏 소스)**: lerobot 0.7.0은 `step`/`smpl`/`ep`를 `format_big_number`로 축약 출력함 (banker's rounding 포함) → step 복원 로직과 `step_exact` 플래그를 추가함
- **발견 (📏 소스)**: `--dataset.eval_split` + `--eval_steps`로 held-out loss를 내장 지원함. `--rename_map`은 fresh init에서 쓸 수 없음
- **E000** (smoke, 200 step): tmux 기록에서 backfill. 2.94 step/s, compute-bound, loss 25.0→3.55
- **상태 정정**: 지침은 E001이 01:50부터 실행 중이라고 가정했으나, 실제로는 미실행이었음 → E001을 `planned`로 등록함
- **데이터**: cohab orange의 episode 길이 long-tail을 확인함 (ep33 2564 frames, ep40 1912 frames)
