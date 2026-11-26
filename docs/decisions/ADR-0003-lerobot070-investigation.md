# ADR-0003: lerobot 0.7.0 조사 메모 — wrist-only 입력, held-out split, ACT 기본값
- 날짜: 2026-11-27 · 상태: Notes (결정은 E002 설계 시 확정)
- 소스: `~/venvs/lerobot/lib/python3.12/site-packages/lerobot/` (0.7.0). 아래 "✅ 확인함"은 소스를 직접 읽은 것이고 **실행 검증은 아님**

## (a) wrist camera만 입력으로 쓰기
- ✅ 확인함: `policies/factory.py:338-341`
  - `output_features`는 항상 dataset에서 다시 만듦
  - `input_features`는 **비어 있을 때만** dataset에서 채움 → 미리 지정하면 그 값이 유지됨
```python
if ds_meta is not None or not cfg.output_features:
    cfg.output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
if not cfg.input_features:
    cfg.input_features = {k: ft for k, ft in features.items() if k not in cfg.output_features}
```
- ✅ 확인함: `--rename_map`은 fresh init에서 쓸 수 없음
  - `configs/train.py:322`에서 `ValueError("rename_map requires a pretrained policy checkpoint")`
  - 따라서 rename으로 front를 제거하는 방법은 불가능함
- ✅ 확인함: `lerobot-edit-dataset --operation.type remove_feature --operation.feature_names "['observation.images.front']"`가 존재함 (`scripts/lerobot_edit_dataset.py:108`)
- 선택지
  1. **파생 dataset (권장 🔮)**: `remove_feature`로 front를 뺀 사본을 만듦
     - 장점: config가 단순하고, front 영상을 디코딩하지 않아 I/O가 줄어듦 (🔮). 데이터 버전 해시로 추적할 수 있음
     - 단점: 디스크 추가 사용 (wrist 영상 + parquet, 수백 MB 🔮)
  2. `--policy.input_features` override: `dict[str, PolicyFeature]`(type, shape)를 draccus CLI나 JSON config로 넘김
     - 🔮 CLI 문법 미검증
     - 🔮 dataset이 여전히 front를 디코딩할 가능성이 있음 (`camera_keys`는 dataset meta 기준, `lerobot_train.py:174`)
- 다음 행동: E002 전에 선택지 1을 짧게(200 step) 스모크 테스트함. 출력 경로와 디스크 사용량도 확인함

## (b) held-out split
- ✅ 확인함: `--dataset.episodes`(list[int])와 `--dataset.exclude_episodes`가 있음 (`configs/default.py:44-46`)
- ✅ 확인함: 더 간단한 내장 방식으로 `--dataset.eval_split=0.1`이 있음 (`datasets/factory.py::make_train_eval_datasets`)
  - task별로 **마지막** ceil(n·split)개 episode를 홀드아웃함 → 이 dataset은 task가 1개이므로 ep 90–99가 eval, 0–89가 train
  - `--eval_steps=N`을 주면 N step마다 `step N: eval_loss=…` 로그가 나옴 → parser가 `offline_eval.jsonl`에 기록함
  - `eval_steps > 0`인데 `eval_split == 0`이면 ValueError (`configs/train.py:360`)
  - `--max_eval_samples`로 eval 비용을 제한할 수 있음
  - streaming dataset과는 함께 쓸 수 없음
- 🔮 주의: 마지막 10 ep가 수집 순서상 편향(숙련도 상승 등)되어 있을 수 있음. random split이 필요하면 `--dataset.episodes`로 직접 지정함
- E002 권장: `--dataset.eval_split=0.1 --eval_steps=2000` (eval 비용은 첫 run에서 측정)

## (c) ACT 기본 hyperparameter (✅ `policies/act/configuration_act.py`)
| 항목 | 기본값 |
|---|---|
| `n_obs_steps` | 1 |
| `chunk_size` | 100 |
| `n_action_steps` | 100 (chunk 전체 실행, temporal ensembling 없음) |
| `temporal_ensemble_coeff` | None |
| `vision_backbone` | resnet18, `ResNet18_Weights.IMAGENET1K_V1` |
| `dim_model` / `n_heads` / `dim_feedforward` | 512 / 8 / 3200 |
| `n_encoder_layers` / `n_decoder_layers` | 4 / 1 |
| `use_vae` / `latent_dim` / `n_vae_encoder_layers` | True / 32 / 4 |
| `dropout` | 0.1 |
| `kl_weight` | 10.0 |
| `optimizer_lr` / `optimizer_lr_backbone` | 1e-5 / 1e-5 |
| `optimizer_weight_decay` | 1e-4 |
| optimizer | AdamW (`get_optimizer_preset`), `grad_clip_norm=10.0` (AdamWConfig 기본) |
| scheduler | 없음 (`get_scheduler_preset → None`, lr 고정) |
| 학습 `seed` (TrainPipelineConfig) | 1000 |
| `log_freq` / `save_freq` 기본 | 200 / 20000 |
- 📏 E000(2 cams)의 파라미터 수: `num_learnable_params=51,597,190` (52M)
- 📏 E000 마지막 줄 기준 $0.676 + 10 \times 0.287 = 3.546$ ≈ `loss` 3.545 → loss = l1 + kl_weight·kld 구조와 일치함
- 📏 E000 `grad_norm` 92–444 ≫ 10 → 사실상 매 step clip이 걸림 (`accelerator.clip_grad_norm_`은 clip 전 total norm을 반환함, `lerobot_train.py:277`). 즉 grad_norm figure는 clip 전 값임
- 🔮 `chunk_size=100` @ 30 fps = 3.3 s open-loop 구간임. 실제 로봇에서는 n_action_steps를 줄이거나 temporal ensembling을 고려할 것 (📚 Zhao et al. 2023, ACT, arXiv:2304.13705)
