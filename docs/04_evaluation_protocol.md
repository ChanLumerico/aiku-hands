# 04. 평가 프로토콜 / Evaluation Protocol
> TL;DR (EN): Success = object(s) in target area within the time limit (default 30 s) with the gripper open and clear.
> Fixed 3×3 start grid (9 positions) shared by sim and real. Sim: 50 rollouts/condition; real: 9×2 = 18
> (fallback 9). Every success rate is reported with its Wilson 95% CI and $n$. Failures get exactly one tag.

## 1. Success 정의
- 제한 시간 $T_{\max}$ (기본 30 s) 안에 물체가 목표 영역(plate) 안에 있음
- 그리고 gripper가 **열린 상태로 물체에서 떨어져 있음**
- pick-orange task는 물체가 3개임 → `n_objects_placed`(0–3)를 함께 기록하고, 주 지표는 **3개 모두 성공**으로 함 (팀 확인 필요 🔮)

## 2. 시작 위치 / Start grid
- 3×3 grid, `position_id` 1–9 (front camera 시점 기준 row-major)
- 실제 테이블에는 테이프로 표시함. sim에서도 같은 9개 좌표를 고정 seed로 씀 → [configs/eval/sim_pick_orange_9grid.yaml](../configs/eval/sim_pick_orange_9grid.yaml)
- 좌표값은 HANDS와 확정 전 (TODO)

## 3. 시행 횟수 / Trials
| 환경 | 조건당 시행 | 구성 |
|---|---|---|
| sim | 50 | 9 위치 순환 + 고정 seed |
| real | 18 (부족하면 9) | 9 위치 × 2 |

- 🔮 $n=18$이면 $\hat p=0.5$에서 Wilson 95% CI 폭이 약 ±0.21임. 조건 간 차이가 0.3 미만이면 real만으로는 구분하기 어려움 → 결론을 쓸 때 반드시 CI가 겹치는지 함께 보고함

## 4. Failure tags (실패 1건당 정확히 1개, 처음 발생한 원인 기준)
| tag | 의미 |
|---|---|
| `perception` | 물체 위치를 잘못 인식 (엉뚱한 곳으로 접근) |
| `grasp_pos` | 접근했지만 grasp 위치·자세 오류 |
| `gripper_timing` | 너무 일찍/늦게 닫거나 엶 |
| `object_drop` | 잡았다가 운반 중 떨어뜨림 |
| `place_pos` | 목표 영역 밖에 놓음 |
| `joint_limit_collision` | joint limit 도달, 테이블·자기 충돌 |
| `stall_jitter` | 멈춤 또는 진동으로 진행 없음 |

- 스키마 검증: `aiku_hands.eval.rollout_schema.RolloutRecord.validate()` (성공이면 tag=None, 실패면 tag 필수)

## 5. 통계 / Statistics
- Wilson score interval ($z = 1.96$):
$$
\tilde p = \frac{\hat p + \frac{z^2}{2n}}{1+\frac{z^2}{n}},\qquad
h = \frac{z}{1+\frac{z^2}{n}}\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}},\qquad
\text{CI} = [\tilde p - h,\ \tilde p + h]
$$
- 구현: [src/aiku_hands/eval/wilson.py](../src/aiku_hands/eval/wilson.py), 표기는 `fmt_rate(k, n)` → `0.667 [0.354, 0.879] (n=9)`
- 집계 결과는 파일로 저장하지 않음. 리포트·figure를 만들 때마다 `rollouts_*.jsonl` 원본에서 다시 계산함
- sim-to-real gap $\Delta = S_{\text{sim}} - S_{\text{real}}$: 두 독립 비율의 차이이므로, CI는 Newcombe hybrid score 방식을 권장함 (📚 Newcombe 1998, *Stat. Med.* 17:873–890) — 구현 예정

## 6. 기록 / Recording
- rollout 1회 = `rollouts_{sim,real}.jsonl` 1줄 (스키마는 [05](05_logging_spec.md))
- real rollout은 영상 파일명 규칙 `W{주차}_{model}_{condition}_p{pos}_t{trial}.mp4` (영상은 git 제외)

## Open questions / 미해결
- 주 지표를 3/3 성공으로 할지, 1개 이상 성공으로 할지 (HANDS와 합의 필요)
- RQ2의 "환경 변화(changed)" 조건 목록 (조명, 배경, 물체 색 등) 미정
