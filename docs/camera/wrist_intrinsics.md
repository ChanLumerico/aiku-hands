# Wrist camera intrinsics (real) / 실측 내부 파라미터
> TL;DR (EN): Real wrist camera calibrated with Zhang's method (17 views, RMS 0.246 px) at 640×480:
> fx=635.90, fy=636.92, cx=347.85, cy=226.30 → HFOV 53.43°, matching the sim template's 53.5° within 0.1°.
> Distortion is strong (k1=−0.441), so sim images (pinhole, no distortion) differ at the borders.

## 1. 결과 / Result (📏 사용자 제공 캘리브레이션 결과, 2026-11)
| 항목 | 값 |
|---|---|
| 해상도 | 640 × 480 |
| 방법 | Zhang (체커보드), 17 views |
| reprojection RMS | 0.246 px |
| $f_x, f_y$ | 635.90, 636.92 |
| $c_x, c_y$ | 347.85, 226.30 |
| dist $(k_1, k_2, p_1, p_2, k_3)$ | −0.44109, 0.2733, −0.00181, 0.00221, −0.06356 |
| HFOV | $2\arctan\!\left(\frac{640}{2 f_x}\right) = 53.43^\circ$ |

$$
K = \begin{bmatrix} 635.90 & 0 & 347.85 \\ 0 & 636.92 & 226.30 \\ 0 & 0 & 1 \end{bmatrix}
$$

## 2. sim과 비교 / vs sim
| 항목 | sim template | real | 차이 |
|---|---|---|---|
| HFOV | 53.5° | 53.43° | 0.07° |
| principal point | (320, 240) 가정 🔮 | (347.85, 226.30) | (+27.9, −13.7) px |
| distortion | 없음 (pinhole) 🔮 | $k_1=-0.441$ (강한 barrel) | 큼 |

- 🔮 principal point 오프셋 약 28 px와 barrel distortion은 sim/real 이미지 차이의 주요 원인이 될 수 있음
- 대응 방안 후보
  - (a) real 이미지를 undistort한 뒤 sim 내부 파라미터로 crop/resample함
  - (b) sim 렌더에 distortion을 적용함
  - (c) DR에 FOV와 principal point 섭동을 포함함

## 3. 재현 / Reproduce
- 캘리브레이션 스크립트와 원본 이미지 위치: 미기록 → 확보 후 여기에 경로를 적을 것

## Open questions / 미해결
- sim 카메라의 정확한 principal point와 distortion 설정 (Isaac Sim USD camera 속성 확인 필요)
- 캘리브레이션 원본 이미지와 스크립트의 보관 위치
