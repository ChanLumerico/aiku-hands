# Wrist camera offset: sim template vs real / 미해결 이슈 추적
> TL;DR (EN): The sim wrist camera uses the LeIsaac template pose on `Robot/gripper`; the cohab dataset appears to
> use the same template (visual match only). The real camera's mount pose is **not measured yet** and visibly
> differs. This is a blocking item for RQ1 with wrist-camera policies.

## 1. sim template (📏 LeIsaac 설정값, 사용자 제공)
- parent: `Robot/gripper`, convention: ROS
- `pos = (-0.001, 0.1, -0.04)` [m]
- `rot_wxyz = (-0.404379, -0.912179, -0.0451242, 0.0486914)`
- HFOV 53.5°

## 2. real (미측정)
- ⚠️ 실제 브래킷 장착 시점은 template과 다름 (육안 확인)
- 측정 방법 후보
  - (a) 브래킷 CAD를 확보해 gripper frame 기준 변환을 계산함
  - (b) 현장 측정: gripper에 체커보드나 AprilTag를 고정하고 hand-eye calibration (AX=XB)
  - (c) 실제 wrist 이미지와 sim 렌더의 정합 최적화

## 3. 영향 / Impact
- 🔮 wrist 시점이 다르면 sim-only ACT의 wrist 입력 분포가 실제와 크게 어긋남 → RQ1 실패 원인이 perception으로 몰릴 수 있음
- 대응
  - (i) real offset을 측정해 sim template을 수정한 뒤 데이터를 재수집하거나 재렌더함
  - (ii) DR에 camera pose 섭동을 포함함 ([configs/dr/dr_v0.yaml](../../configs/dr/dr_v0.yaml))

## 4. 상태 로그 / Status log
| 날짜 | 상태 |
|---|---|
| 2026-11-27 | 미측정. CAD 확보 또는 현장 측정 필요 (HANDS 요청) |
