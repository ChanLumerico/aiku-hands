# ADR-0001: 연구 방향을 Sim-to-Real ACT + DR로 전환 / Pivot to Sim-to-Real ACT + DR
- 날짜: 2026-11 · 상태: Accepted

## Context
- 이전 방향은 latent action learning이었음
- 일정(W6–W11)과 하드웨어(SO-101 한 대)를 고려할 때, 측정 가능한 결과를 확보하는 것이 우선임

## Decision
- sim(Isaac Lab/LeIsaac) 데이터만으로 ACT를 학습하고, 실제 SO-101에서 평가함 (RQ1)
- 같은 조건에서 DR 유무만 바꿔 robustness를 비교함 (RQ2). 선택적으로 소량 real demo로 fine-tune함 (RQ3)

## Consequences
- 평가 프로토콜(9-grid, Wilson CI, failure tag)을 sim과 real에서 동일하게 써야 함 → [04](../04_evaluation_protocol.md)
- 카메라 정합(내부 파라미터, wrist offset)이 핵심 리스크가 됨 → [camera/](../camera/)
- 상세 경위는 미기록. 필요하면 사용자 확인 후 보충함
