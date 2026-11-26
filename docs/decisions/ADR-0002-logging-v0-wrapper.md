# ADR-0002: 학습 로깅 v0 = lerobot-train subprocess wrapper
- 날짜: 2026-11-27 · 상태: Accepted

## Context
- lerobot 0.7.0 내장 추적기는 trackio임. 오프라인·재현용 원본으로는 JSONL이 필요함 (figure를 다시 그리기 위해)
- 학습 루프를 수정하면 결과가 바뀔 위험이 있음

## Decision
- v0은 학습 코드를 건드리지 않음. stdout 파싱 + psutil sysmon + manifest로 구성함 ([05](../05_logging_spec.md))
- v1(in-process hook)은 같은 seed로 loss 일치를 검증한 뒤에만 도입함

## Consequences
- log_freq 창 평균만 기록됨 (step별 값은 없음). MPS allocator 메모리는 v0에서 `null`임
- `step` 축약 출력(`14K`)은 복원 로직과 `step_exact` 플래그로 보완함
