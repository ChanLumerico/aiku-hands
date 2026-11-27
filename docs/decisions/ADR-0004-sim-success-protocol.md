# ADR-0004: sim success = 프로젝트 프로토콜 정의 (LeIsaac `task_done` 대신)
- 날짜: 2026-11-28 · 상태: **Accepted** (사용자 승인 2026-11-28 05:37 KST)
- 관련: docs/04 (평가 프로토콜), CLAUDE.md §1.4, docs/08 §5, `sim/leisaac_act_rollout.py` (`ProtocolJudge`)

> TL;DR (EN): LeIsaac's `task_done` additionally requires the arm inside a rest-pose box (±30°/joint) that only
> 18/100 dataset demos end in, so 31/120 sweep-1 episodes that placed and left all three oranges on the plate were
> scored as failures. Sim `success` now follows the project protocol (objects in target, released, held 1 s,
> within 30 s); LeIsaac `task_done` is kept as the secondary field `success_leisaac`.

## 1. 배경 / Context
- 📏 LeIsaac 0.4.0 `tasks/pick_orange/mdp/terminations.py::task_done`의 조건 (HANDS 소스 확인)
  - 오렌지 3개 모두 접시 기준 x·y ±10 cm, z ±7 cm
  - **`is_so101_at_rest_pose`**: 6개 관절이 모두 rest center ±30° 안 (pan 0, lift −100, elbow 90, wflex 50, wroll 0, gripper −10)
- 📏 학습 데이터 (`scripts/analyze_dataset_rest_pose.py`, cohab_orange 100 ep)
  - 마지막 frame이 rest box 안인 ep: **18/100**
  - 벗어난 관절: wroll 77, pan 48, gripper 12 (lift 1, elbow·wflex 0)
  - → 데모는 "접힌 자세"로 끝나지만 wrist_roll·pan이 LeIsaac 기준 밖임
- 📏 sweep 1 (`experiments/E001_act_orange_2cam_b8/metrics/rollouts_sim.jsonl`, 6 조건 × 20 ep)
  - 오렌지 3개가 마지막 확인 시점까지 접시 안에 있었는데 `task_done`이 false인 ep: **31/120** (h50·h25 각 8/20)
  - 영상 확인 (70k h50 trial 2): 약 10 s에 3개 배치 → 팔을 접고 45 s 이상 정지
- 프로젝트 프로토콜(CLAUDE.md §1.4)은 rest 복귀를 요구하지 않음: "제한 시간 안에 물체가 목표 영역에 있고, gripper가 열린 채로 물체에서 떨어져 있음"

## 2. 결정 / Decision
`success_def = protocol_v1` (기본값)
1. 오렌지 3개 모두 접시 box 안 (x·y ±10 cm, z ±7 cm; `task_done`과 동일)
2. **released**: robot의 `gripper`·`jaw` body origin에서 모든 오렌지 중심까지 최소 거리 > `release_dist` = 0.10 m
3. 1·2를 **60 step (1 s) 연속** 만족
4. 정책 시작 후 **1,800 step (30 s)** 안 (warmup 제외)
- LeIsaac `task_done`이 먼저 발생하면 success로 셈 (오렌지 조건 동일 + 더 엄격한 로봇 조건). 별도 필드 `success_leisaac`로 항상 기록함
- 성공 즉시 episode 종료 → `env.reset()` 1회 (auto-reset과 같은 `inference_mode` 안). episode당 reset 1회라 같은 seed에서 trial별 초기 배치가 조건 간 동일함 → `init_layout`으로 검증
- 기록 필드: `success_def`, `success_protocol`, `success_leisaac`, `end_reason`, `success_step`, `first_all_placed_step`, `release_check`, `min_grip_orange_dist_m`, `success_grip_orange_dist_m`, `init_layout`

## 3. 프로토콜 문구와의 차이 / Deviations (🔮)
- "gripper가 열린 채로": sim에서는 **gripper 개도 대신 "물체와 떨어짐(released)"만 확인**함
  - 데모가 gripper를 닫은 채 접힌 자세로 끝나는 경우가 있음 (gripper 위반 12/100) → 개도 조건을 넣으면 같은 종류의 불일치가 다시 생김
  - 의도("물체를 놓고 떨어짐")는 거리 조건이 직접 측정함
- 거리 기준은 fingertip이 아니라 **body origin**임. 0.10 m는 검증 전 값이므로 `min_grip_orange_dist_m` 분포로 사후 점검함
- real 평가(RQ1)도 같은 정의를 써야 sim-to-real gap이 의미 있음 → real 판정 절차서에 반영 필요 (HANDS)

## 4. 결과 / Consequences
- sweep 1 수치(LeIsaac 기준)는 그대로 보존함. 새 정의 기준 재측정은 sweep 2 (`configs/eval/E001_eval2_exec_warmup.json`)
- 성공 ep가 일찍 끝나서 sweep 시간이 줄어듦
- 사후 근사: 과거 record에서는 `all3_end` (마지막 확인 시점 오렌지 3개 in-box; release 미확인)를 protocol_v1 proxy로 씀 (`scripts/analyze_sim_rollouts.py`)
