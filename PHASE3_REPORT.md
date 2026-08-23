# Phase 3 Baseline Implementation Report

- 완료일: 2026-08-23 (Asia/Seoul)
- 범위: Phase 3 baseline 구현 및 synthetic/소규모 smoke 검증
- 기준 상태: 저장되어 있던 43개 테스트 통과 구현에서 이어서 보강
- 실행하지 않은 작업: `v1_legacy` 전체 학습, baseline test split 평가,
  Cell-MSCA 및 attention ablation 구현

## 구현 결과

### 공통 baseline 계약

다음 다섯 baseline이 `BaselineDataProtocol`, train-cell 전처리 통계,
validation-only tuner 및 공통 evaluator를 사용하도록 구성했다.

1. train-mean
2. raw-target LightGBM
3. log1p-target LightGBM
4. Tweedie LightGBM
5. Concat-MLP

모든 논문용 result row는 data/split/split-config/preprocessing/model-config
SHA-256, split/train seed, target transform/scale, objective, inverse mode,
smearing factor, original-unit headline metrics, log-space secondary metrics,
secondary Spearman 및 best iteration/epoch를 기록한다. Hotspot 및 분류 metric은
result schema에서 허용하지 않는다.

### LightGBM 호환 경로

`LGBMRegressor.fit` signature를 검사하여 다음 두 API를 분리했다.

- LightGBM 4.7: 단일 배열 `eval_X`, `eval_y`
- LightGBM 4.0–4.6 호환 경로: 단일 validation tuple을 담은 `eval_set`

두 경로 모두 evaluation name은 `validation` 하나이며 test 배열을 받지 않는다.
log1p LightGBM의 early-stopping metric은 median inverse 후의 validation
original-unit MAE이다.

### PyTorch Concat-MLP

기존 NumPy log-MSE 구현을 제거하고 `torch`를 함수 호출 시점에만 import하는
PyTorch 구현으로 교체했다.

- 허용 loss: `log_l1`, `log_huber`
- Huber delta: `huber_delta`로 config 및 config SHA-256에 포함
- optimizer: AdamW
- weight decay: weight parameter group에만 적용, bias group은 `0.0`
- dropout: `dropout`으로 config 및 config SHA-256에 포함
- checkpoint: median inverse validation original-unit MAE만 사용
- trainer 재사용: `fit_log_neural_model()`이 model factory를 받아 이후
  Cell-MSCA trainer에서 재사용 가능

현재 실행 환경에는 PyTorch가 없었다. 사용자 제약에 따라 설치하지 않았고,
지연 import 및 명확한 dependency 오류 경로와 optimizer parameter-group 계약을
테스트했다. 실제 PyTorch 학습 smoke는 수행하지 않았다.

### log 모델 선택 순서

모든 log 모델에 다음 순서를 고정했다.

1. median inverse validation original-unit MAE로 checkpoint 또는 early stopping
2. 선택된 checkpoint 복원
3. train residual만으로 Duan factor 계산
4. 같은 validation 예측에서 median과 Duan original-unit MAE 비교
5. inverse mode와 smearing factor를 동결한 뒤에만 test authorization 허용

### Validation 산출물

실행 config에 `output_dir`을 추가했다. Validation tuning runner는 다음 파일을
자동 저장하며 동일 경로의 기존 파일이 하나라도 있으면 전체 저장을 거부한다.

- `candidate_results.json`
- `selected_results.json`
- `<model>.candidate_<index>.validation_predictions.csv`
- `validation_run_manifest.json`

각 prediction CSV 저장 직후 original/log metric과 Spearman을 CSV만으로 다시
계산하여 result row와 비교한다. 검증 통과 상태는 candidate JSON 및 manifest에
기록된다. Manifest는 `test_subset_materialized: false`를 기록한다.

### Frozen test gate

`create_frozen_baseline_selection()`이 validation 선택 결과만으로
`FROZEN_BASELINE_SELECTION.json`을 생성한다. 파일에는 공통 data/split/
split-config/preprocessing hash와 seed, 모델별 config hash, target/inverse 계약,
smearing factor, best iteration/epoch 및 validation metric을 기록한다.

`authorize_test_evaluation()`은 frozen 파일의 존재 여부, schema, 모든 공통 hash,
seed, 모델 config/inverse/best iteration 또는 epoch/validation metric 일치를 먼저
검증한다. 파일이 없거나 값이 다르면 test data를 materialize하기 전에
`TestEvaluationBlockedError`를 발생시킨다. 이번 작업에서는 frozen 생성 및
authorization 검증까지만 테스트했고 test split을 baseline 평가에 열지 않았다.

### 이식 가능한 예시 config

`configs/baseline_example.json`의 임시 작업 폴더 참조를 제거했다.

- data glob: `data/v1_legacy/*.npz`
- split: `splits/cell_fixed_seed42.csv`
- metadata: `splits/cell_fixed_seed42.metadata.v2.json`
- output: `outputs/phase3_baselines/example_validation`

현재 checkout에는 `data/v1_legacy` 디렉터리가 없다. 실제 로컬 archive 위치를
사용할 때는 tracked 예시를 바꾸지 않고 별도의 untracked config 사본에서
`npz_glob`과 필요 시 `output_dir`을 override해야 한다.

루트 `IMPLEMENTATION_SPEC.md`는 공급된 원본을 그대로 포함했다. 원본과 프로젝트
사본의 SHA-256은 모두
`6AC8A4462FA99B6CA21BFF7ECF8712CBB13428CA141F7F65B77D865A9BD4697D`이다.

## 실제 LightGBM 4.7 smoke

프로젝트 밖 임시 폴더에만 `lightgbm==4.7.0`과 smoke 의존성을 설치하고
`tests/lightgbm47_smoke.py`를 실행했다.

| Model | Prediction shape | Best iteration | Eval data | Fit API |
|---|---:|---:|---|---|
| `lightgbm_raw` | `[32]` | 40 | `validation` only | single `eval_X`/`eval_y` |
| `lightgbm_log1p` | `[32]` | 40 | `validation` only | single `eval_X`/`eval_y` |
| `lightgbm_tweedie` | `[32]` | 40 | `validation` only | single `eval_X`/`eval_y` |

- 실제 버전: LightGBM 4.7.0
- 데이터: synthetic train 96 samples / validation 32 samples
- test split 생성 또는 평가: 없음
- 임시 설치 폴더 정리: 통과, 잔여 폴더 0개
- 정리 후 전역/프로젝트 import 상태: LightGBM 없음

LightGBM 4.0–4.6 `eval_set` 경로는 fake estimator 회귀 테스트로 유지했으며,
해당 구버전 패키지들의 실제 설치 smoke는 수행하지 않았다.

## 검증 결과

### 전체 테스트

- 명령: `python -m unittest discover -s tests -v` (`PYTHONPATH=src`)
- 결과: **48/48 통과**
- 저장 기준 43개 테스트: 모두 유지
- 신규 회귀 테스트: 5개
  - frozen selection data hash mismatch 차단
  - log LightGBM checkpoint metric의 median inverse 검증
  - PyTorch dependency 오류 및 MLP config 계약
  - AdamW weight/bias decay group 분리
  - validation artifact 저장, metric/Spearman 재계산 및 no-overwrite

### Compileall 및 Ruff

- `python -m compileall -q src tests`: 통과, exit code 0
- Ruff: 현재 환경에서 module/executable을 찾을 수 없어 미실행
- PyTorch 실제 smoke: 미실행; 현재 환경에 PyTorch 없음, 임의 설치하지 않음

## 변경 파일

이번 continuation에서 변경 또는 추가한 파일:

- `src/cell_msca/__init__.py`
- `src/cell_msca/baselines.py`
- `src/cell_msca/neural_baselines.py`
- `src/cell_msca/experiment.py`
- `tests/test_baselines.py`
- `tests/lightgbm47_smoke.py`
- `configs/baseline_example.json`
- `IMPLEMENTATION_SPEC.md`
- `PHASE3_REPORT.md`

저장되어 있던 Phase 3 구현에서 이미 변경되어 이번 결과에 포함된 파일:

- `src/cell_msca/metrics.py`
- `src/cell_msca/evaluate.py`

기존 flat `src/*.py`, 기존 split, result 및 checkpoint는 수정·삭제·덮어쓰지
않았다.

## Flat source SHA-256 불변 검증

`PHASE2_1_REPORT.md`의 기준 hash와 현재 flat `src/*.py` 11개를 다시 비교했다.
불일치 수는 0개다.

| 파일 | SHA-256 | 상태 |
|---|---|---|
| `baselines.py` | `4D085A58D4659A7B72C2154E99DA51D4563B8DEB8CC45CBDE90F60FF6725685D` | 불변 |
| `diagnostics.py` | `6DC7E9CAFE1BD191E986959255504D1C6604714B0446B18DCC8C18F7F9B3F402` | 불변 |
| `evaluation.py` | `CC965DCCADBB66276EB70325485D7D85D4F46AD359BD21665BB0335707268194` | 불변 |
| `fast_data.py` | `4C111C9107250CED96121FD34C978D8FF3755A4EE76769C855A44EBFA0EF70C9` | 불변 |
| `inference.py` | `DD0E31FF268CEDEBCFD1AA68D965CBDDA7ACCDC537E8B1DAAFF1E1F6F74A6526` | 불변 |
| `make_figures.py` | `56755228753621AAD7A6C94035FE7CE77690183DC058F3E58F4525E520DB9642` | 불변 |
| `metrics.py` | `022238944E8940B1850FFA2A06320EF9674DB55759910D9F901F48B00DBE3A5D` | 불변 |
| `model.py` | `29957A88665F687CA4770A866AD87C279A2AA71CFE07F9EA78D5A573C2ED1245` | 불변 |
| `spatial_grid_dataset.py` | `AB078C55A7349D18379FFDFA7F2175EEEFEE823D785446A3F803C00D486AF29C` | 불변 |
| `splits_v2.py` | `EA6350CF8BD36B2E0D52A959B058741FB0DF20C368F17920B680714233625BB2` | 불변 |
| `train.py` | `8181117532F01E765192C759496BF7584751D582E51428E505FBAC348CC3A9C1` | 불변 |

## Phase 3 완료 조건 판정

| 완료 조건 | 판정 | 근거 |
|---|---|---|
| 다섯 baseline 구현 | 통과 | train-mean, raw/log1p/Tweedie LightGBM, lazy-PyTorch Concat-MLP |
| 모든 모델이 동일 persistent split 사용 | 통과 | 단일 `BaselineDataProtocol`, hash 일관성 회귀 테스트 |
| provenance와 공통 result schema | 통과 | evaluator schema 및 48개 테스트 |
| validation-only tuning/early stopping | 통과 | 구형·신형 API 테스트 및 실제 4.7 smoke |
| log checkpoint의 median inverse MAE | 통과 | LightGBM 회귀 테스트 및 공통 neural trainer 구현 |
| checkpoint 후 Duan 비교/train residual only | 통과 | 호출 순서 구현 및 residual 배열 회귀 테스트 |
| inverse mode test 전 동결 | 통과 | frozen JSON 검증 없이는 authorization 실패 |
| validation 산출물 저장/no-overwrite/CSV 검증 | 통과 | synthetic 통합 테스트 |
| Spearman secondary only | 통과 | result schema와 prediction round-trip 검증 |
| 실제 LightGBM 4.7 호환성 | 통과 | 세 변형 actual smoke |
| 실제 PyTorch 학습 smoke | 환경상 미실행 | PyTorch 미설치, 설치 금지 지침 준수 |
| `v1_legacy` 전체 학습 및 test 평가 | 의도적으로 미실행 | 이번 단계의 명시적 제약 |

## 아직 미검증인 사항

- 실제 PyTorch가 설치된 환경에서 Concat-MLP forward/backward/checkpoint smoke
- 실제 `v1_legacy` 자료를 사용한 다섯 baseline validation run과 산출물 생성
- LightGBM 4.0–4.6 각각의 실제 패키지 smoke; 호환 분기 단위 테스트만 수행
- frozen selection 이후의 실제 test 평가는 의도적으로 수행하지 않음
- `v1_legacy` target의 물리 단위 및 과거 전처리 한계는 Phase 2 보고와 동일
