# 00. 프로젝트 개요 / Project Overview
> TL;DR (EN): Train ACT purely on Isaac Lab (LeIsaac) simulation demos, deploy on one real SO-101, and test whether
> sim-side Domain Randomization (DR) improves real-world robustness. Three RQs, three models (A nominal, B DR,
> C = B + few real demos), four milestones (W6–W11).

## 1. 목표 / Goal
- Isaac Lab(LeIsaac) sim 데이터만으로 ACT를 학습한 뒤 실제 SO-101 한 대에서 pick-and-place를 수행함
- sim DR이 실제 환경 변화에 대한 robustness를 높이는지 검증함
- 이전 방향(latent action learning)은 2026-11에 폐기함 → [ADR-0001](decisions/ADR-0001-sim-to-real-act-dr.md)

## 2. Research Questions
| ID | 질문 | 지표 |
|---|---|---|
| RQ1 | sim-only ACT가 실제 SO-101에서 동작하는가 | $\Delta_{\text{s2r}} = S_{\text{sim}} - S_{\text{real}}$ |
| RQ2 | sim DR이 실제 성능·환경 변화 robustness를 높이는가 | 조건별 $S$, $\text{drop} = S_{\text{normal}} - S_{\text{changed}}$ |
| RQ3 (선택) | 소량 real demo 추가 효과 | real 10/20/30 ep fine-tune 비교 |

- $S$는 항상 Wilson 95% CI와 $n$을 함께 보고함 ([04](04_evaluation_protocol.md))

## 3. 모델 / Models
| 모델 | 학습 데이터 | 비고 |
|---|---|---|
| A (nominal) | 고정 sim 환경 | 현재 후보 dataset: `edge-inference/cohab-so101-pick-orange-dataset` |
| B (robust) | DR sim 데이터 | episode 수·hyperparameter는 A와 동일 |
| C10/C20/C30 | B + real 10/20/30 ep fine-tune | RQ3 |

## 4. 마일스톤 / Milestones
| ID | 주차 | 내용 | 상태 (2026-11-27) |
|---|---|---|---|
| M1 | W6 | sim pipeline 동작 | 진행 중 — 학습 pipeline 스모크 통과 (E000), sim rollout 미착수 |
| M2 | W7 | sim-to-real 수치 (RQ1) | 대기 — 실제 wrist offset 미측정, 모터 보드 USB 오류 |
| M3 | W10 | nominal vs robust 실제 비교 (RQ2, 주목표) | 대기 |
| M4 | W11 | sim+real 비교 (도전) | 대기 |

- 일정이 2주 이상 밀리면 W11을 빼고 RQ1·RQ2에 집중함

## 5. 역할 / Who does what
- **AIKU (학습)**: Mac Studio에서 ACT 학습, 로깅, 오프라인 평가, 리포트
- **HANDS (하드웨어·sim)**: HANDS PC의 Isaac/LeIsaac sim rollout, 실제 SO-101 정비·평가, 카메라 장착

## Open questions / 미해결
- Model A의 학습 데이터가 외부 dataset(cohab, 100 ep teleop)으로 충분한지, 아니면 자체 수집(9-grid 정렬)이 필요한지
- DR 범위([configs/dr/dr_v0.yaml](../configs/dr/dr_v0.yaml))와 B용 데이터 수집 방법(재렌더 vs 재수집)
