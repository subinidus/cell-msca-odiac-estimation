# Phase 5A Validation Robustness 구현 보고서

작성일: 2026-09-11

branch: `experiments/phase-5a-validation-robustness`

기준 commit: `034712aff6670b0fdb8b272e7b9311dc9b652830`

## 1. 범위와 결과

Phase 5A는 validation-only 추가 실험을 실행하고 검증하는 코드와 동결 계약만
구현했다. 실제 `v1_legacy` 전체 학습, seed 42 신경망 재학습, test subset
materialization 또는 평가는 수행하지 않았다. 모델 architecture, Q/K/V, target,
feature order, persistent split 및 metric 정의도 변경하지 않았다.

구현된 실행 순서는 다음과 같다.

1. archive 또는 expanded 36-NPZ input 무결성 확인;
2. 생성 데이터를 사용하는 기존 네 Cell-MSCA variant smoke;
3. raw/log1p LightGBM seed 42 convergence diagnostic;
4. validation original-unit MAE가 낮은 LightGBM 후보를 파일로 동결;
5. 선택된 LightGBM과 bidirectional/token-no-attention의 seed 43/44 반복;
6. 기존 seed 42 package와 새 결과의 provenance 및 prediction 재검증;
7. 3-seed metric 요약과 paired validation-cell cluster bootstrap.

모든 project-data training command는 기본적으로 닫혀 있으며
`--allow-full-validation`을 명시해야 한다. CLI에는 test 실행 command가 없다.

## 2. 데이터 및 split 계약

고정 provenance는 다음과 같다.

| 항목 | SHA-256 |
| --- | --- |
| data | `c3a1889fd863117c3faa9125094267974dbacfe6fd4ca6c157dead2d7c874b62` |
| split assignment | `8b795ab11ad741a5e161b08ee1c2d44cefa1b21e5b6a0a724a7ca88a85cacb40` |
| split config | `b790de004c019af6842da99723c7d608d6a5bfa454d70b3053f26eca4287cb8b` |
| preprocessing | `35f760ab7a73ae561ae42562889e67cd1ac1143cdfa7ab6a5b3b25352efcc4db` |
| archive | `5e01b3bc58fa2adebbe24aa2b6c3d2d2b93d274adc1c2686a37c62f57a5d93ed` |

`verified_v1_input()`은 `/kaggle/input` 아래에서 다음 중 정확히 하나만 허용한다.

- `2022_2024_npz.zip` 하나: working 임시 directory에 안전하게 풀고 context 종료 시
  정리;
- 36개 frozen NPZ가 들어 있는 directory 하나: 압축 해제 없이 사용.

두 경로 모두 파일명, 크기, file SHA-256, canonical NPZ content SHA-256을 frozen
archive manifest와 대조한다. expanded directory가 여러 개이거나 archive와
expanded directory가 동시에 발견되면 fail closed한다. archive의 file hash와
NPZ content hash는 서로 다른 provenance 항목으로 유지한다.

dataset을 만든 뒤 기존 metadata-bearing `cell_fixed_seed42` split을 안전 로드하고
data/split/split-config hash를 확인한다. preprocessing은 train cell로만 다시 fit한다.
test assignment 수와 hash는 metadata 검증 과정에서 확인하지만 test sample array는
materialize하지 않는다. integrity summary와 run/aggregation manifest에는
`test_sample_arrays_materialized=false` 또는 `test_subset_materialized=false`를
기록한다.

연구상 제한도 config에 고정했다. `label_reg`는 `v1_legacy ODIAC-derived zonal
mean`이며 직접 측정값, target-cell 총량 또는 mass-conserving target이라고 주장하지
않는다. 정확한 물리 단위는 미해결이다. archive feature에는 split 이전 월별
whole-grid imputation이 이미 포함되어 있어 현재 train-only preprocessing으로
되돌릴 수 없다.

## 3. LightGBM convergence와 선택 gate

기존 raw/log1p 결과의 `best_iteration=1000`이 이전 상한과 같았으므로 두 후보에만
다음 진단 설정을 적용했다.

| 항목 | 기존 | Phase 5A |
| --- | ---: | ---: |
| `n_estimators` | 1,000 | 5,000 |
| `early_stopping_rounds` | 50 | 200 |

learning rate `0.03`, `num_leaves=31`, min-child, row/column subsampling, regularization,
objective 및 deterministic seed 조건은 유지한다. LightGBM 공식
[Parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html),
[Python guide](https://lightgbm.readthedocs.io/en/stable/Python-Intro.html),
[early-stopping callback](https://lightgbm.readthedocs.io/en/v4.6.0/pythonapi/lightgbm.early_stopping.html)은
boosting-round 상한과 validation metric 기반 early stopping의 동작 근거다. 정확한
5,000/200은 공식 권장값이 아니라 이전 상한 도달을 진단하기 위해 결과 확인 전에
고정한 프로젝트 설정이다.

early stopping은 validation만 받고 original-unit MAE를 사용한다. log1p 후보의
checkpoint metric은 median inverse original-unit MAE다. checkpoint 확정 후에만
train residual로 Duan factor를 계산하고 validation에서 median과 Duan을 비교한다.
각 manifest는 상한, patience, best iteration 및 상한 도달 여부를 기록한다.

두 convergence artifact의 prediction metric, hashes, Git SHA와 config SHA를 다시
검증한 뒤 `FROZEN_PHASE5A_LIGHTGBM_SELECTION.json`을 생성한다. 선택 기준은 validation
original-unit MAE이며 exact tie는 사전 순서 raw, log1p로 처리한다. 이 파일이 없거나
변경되면 seed 43/44 반복을 거부한다. Tweedie는 실행 대상이 아니다.

## 4. 3-seed 집계

train seed는 `[42, 43, 44]`, split seed는 `42`로 고정했다. seed 42 neural 결과는
기존 package만 사용한다. selected LightGBM의 3-seed 비교에서는 새 convergence
설정으로 얻은 seed 42 artifact와 seed 43/44 artifact를 사용해 model config를
일치시킨다.

집계 전에 다음을 확인한다.

- 기존 seed 42 package: 네 reference MAE의 소수점 6자리 값, base Git SHA, data,
  split, split-config, preprocessing 및 모델 config SHA;
- 새 결과: Phase 5A Git SHA, 공통 provenance, seed별 model config SHA;
- 모든 파일: prediction CSV에서 original/log metric과 Spearman exact 재계산;
- 모델과 seed 사이: row order, `cell_id`, true original/log target exact 일치.

각 모델의 original-unit MAE, RMSE, R², bias와 Spearman에 대해 세 seed의 mean 및
sample standard deviation(`ddof=1`)을 기록한다. log-space metric은 개별 공통 result
schema의 secondary metric으로 유지한다.

## 5. Paired cell-cluster bootstrap

각 모델의 세 seed original-unit prediction을 row 단위로 평균한 다음 paired MAE
difference를 계산한다.

- 비교 1: bidirectional 대 token-no-attention;
- 비교 2: bidirectional 대 선택된 LightGBM;
- 방향: `MAE(bidirectional) - MAE(comparator)`;
- resampling unit: validation 고유 `cell_id`;
- cluster: 선택된 cell의 36개월 행 전체;
- 반복: 10,000;
- bootstrap seed: 3407;
- interval: 95% percentile CI.

모든 cell이 정확히 36행을 갖지 않으면 실행을 거부한다. 이 bootstrap은 frozen
3-seed 평균 prediction에 조건부인 validation-cell sampling 불확실성을 나타내며,
training-seed 불확실성 자체를 bootstrap하지는 않는다.

## 6. 생성·변경 파일

- `src/cell_msca/phase5a.py`: validation-only CLI, convergence/selection/repeat,
  artifact 검증, 3-seed 집계 및 paired cluster bootstrap;
- `src/cell_msca/v1_validation.py`: archive와 verified expanded input의 공통
  read-only 검증 경로;
- `configs/phase5a_validation_robustness.json`: master frozen contract;
- `configs/phase5a_cell_msca_seed43.json`;
- `configs/phase5a_cell_msca_seed44.json`;
- `configs/phase5a_experiment_assignments.json`;
- `docs/PHASE5A_KAGGLE_RUNBOOK_KO.md`;
- `tests/test_phase5a.py`;
- `pyproject.toml`: `cell-msca-phase5a` console entry point;
- `PHASE5A_REPORT.md`.

NPZ, ZIP, checkpoint, prediction, 실제 result artifact 또는 persistent split은
생성·수정·커밋 대상에 포함하지 않았다.

## 7. 검증 결과

- 신규 Phase 5A tests: 14개 통과;
- 전체 stored tests: 103개 통과, skip 0, error 0;
- 실제 CPU synthetic runner smoke: token-no-attention, forward, reverse,
  bidirectional 모두 완료; 각 7개 표준 artifact 생성과
  `test_subset_materialized=false` 확인;
- 검증 환경: Python 3.14.0, NumPy 2.5.2, PyTorch 2.13.0+cpu,
  LightGBM 미설치;
- `compileall src tests`: 통과;
- Ruff: 검증 환경에 설치되어 있지 않아 미실행;
- `git diff --check`: 통과;
- flat legacy `src/*.py` 11개: 기준 SHA-256과 모두 일치, mismatch 0개.

synthetic artifact는 OS 임시 directory에만 생성했으며 context 종료 시 정리됐다.
실제 project data 학습 및 test 평가는 실행하지 않았다.

## 8. 실행 전 남은 사항

1. Kaggle에 기존 seed 42 result package와 private v1 archive 또는 expanded NPZ
   directory를 첨부하고 integrity-only 검증을 통과해야 한다.
2. Kaggle 환경의 actual LightGBM/PyTorch/CUDA 버전과 dependency compatibility를
   확인해야 한다.
3. raw/log1p convergence 결과는 아직 없으므로 선택된 LightGBM과 상한 도달 여부도
   아직 미정이다.
4. seed 43/44 실제 validation 결과, 3-seed 요약 및 bootstrap CI는 아직 생성하지
   않았다.
5. test gate는 계속 닫혀 있다. Phase 5A 결과만으로 test 평가를 승인하지 않는다.
