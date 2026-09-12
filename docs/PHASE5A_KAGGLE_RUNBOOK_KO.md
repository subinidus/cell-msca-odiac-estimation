# Phase 5A validation robustness Kaggle 실행 지침

## 1. 범위와 고정 계약

Phase 5A는 test gate를 열지 않는 추가 validation 실험이다. 기존 seed 42 전체
validation을 다시 실행하지 않는다. 예외는 이전 1,000 round 상한에 도달한 raw와
log1p LightGBM의 seed 42 convergence diagnostic뿐이다.

- persistent split seed: `42`;
- 사전 고정 train seed: `42`, `43`, `44`;
- 새 신경망 학습: bidirectional 및 token-no-attention의 seed `43`, `44`;
- 새 LightGBM 학습: raw/log1p seed 42 convergence check 후 선택된 하나의 seed
  `43`, `44`;
- 제외: forward, reverse, concat MLP, Tweedie 재실행;
- 선택 기준: validation original-unit MAE;
- headline: original-unit MAE, RMSE, R², bias;
- secondary: log-space metric 및 Spearman;
- test subset: assignment 무결성만 검사하며 sample array를 materialize하지 않는다.

`v1_legacy` target은 기존 ODIAC-derived zonal mean이다. 직접 측정값, target-cell
총량 또는 mass-conserving target으로 해석하지 않는다. 물리 단위도 이 단계에서
확정하지 않는다. archive에 포함된 feature에는 persistent split 이전의 월별
whole-grid 결측치 대체가 이미 적용되어 있으며, 현재 train-only preprocessing은
이를 되돌리지 못한다.

## 2. LightGBM convergence 설정 근거

기존 raw/log1p 실행은 모두 `best_iteration=1000`으로 이전 상한에 도달했다. Phase
5A는 learning rate `0.03`, `num_leaves=31` 및 나머지 조건을 그대로 두고 다음 두
값만 사전에 변경한다.

| 항목 | 기존 | Phase 5A |
| --- | ---: | ---: |
| `n_estimators` 상한 | 1,000 | 5,000 |
| `early_stopping_rounds` | 50 | 200 |

LightGBM 공식 [Parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html),
[LGBMRegressor API](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRegressor.html),
[early-stopping callback](https://lightgbm.readthedocs.io/en/v4.6.0/pythonapi/lightgbm.early_stopping.html)
문서는 boosting-round 상한과 validation metric 기반 early stopping의 동작을
근거로 한다. 5,000/200이라는 수치는 공식 권장값이 아니라 이전 상한 도달을
확인하기 위한 프로젝트 고유의 진단 설정이며, 새 결과를 보기 전에 동결한다.
각 validation metrics artifact와 manifest는 최적 점수 시점인 `best_iteration`과
실제 수행 횟수인 `actual_iterations`를 구분해 기록한다. 상한 도달 여부는 공식
sklearn API의 `n_estimators_`/`n_iter_` 또는 Booster fallback으로 얻은
`actual_iterations`를 기준으로만 계산한다.

log1p 모델의 early stopping metric은 median inverse validation original-unit MAE다.
checkpoint가 정해진 뒤 train residual만으로 Duan factor를 계산하고, validation에서
median/Duan original-unit MAE를 비교한다. 선택된 inverse는 해당 prediction
artifact에 고정된다.

## 3. Kaggle input 구성

public repository에서 exact Phase 5A commit을 checkout하는 방식을 사용한다. attached
code mode는 사용하지 않는다. Kaggle private Dataset에는 다음을 첨부한다.

1. `2022_2024_npz.zip` 하나 또는 그 archive에서 풀린 36개 NPZ directory 하나;
2. 기준 SHA `034712aff6670b0fdb8b272e7b9311dc9b652830`에서 생성된 기존 seed 42
   validation result package;
3. raw data, credential 또는 token은 포함하지 않는다.

runner는 `/kaggle/input` 아래에서 archive와 expanded input 중 정확히 한 종류만
허용한다. archive가 있으면 Kaggle working 임시 directory에 안전하게 풀고 종료 시
정리한다. expanded input이면 36개 각각에 대해 frozen file name, byte size, file
SHA-256, canonical NPZ content SHA-256을 모두 확인한다. archive 지원은 유지된다.
두 종류가 함께 있거나 후보가 둘 이상이면 중단한다.

## 4. source와 환경 준비

아래 `<PHASE5A_COMMIT_SHA>`는 이 branch의 검토된 commit SHA로 교체한다.

```bash
export PHASE5A_SHA=<PHASE5A_COMMIT_SHA>
export REPO=/kaggle/working/cell-msca-source
git clone https://github.com/subinidus/cell-msca-odiac-estimation.git "$REPO"
git -C "$REPO" checkout --detach "$PHASE5A_SHA"
test "$(git -C "$REPO" rev-parse HEAD)" = "$PHASE5A_SHA"
export PYTHONPATH="$REPO/src"
python -m pip check
python -m unittest discover -s "$REPO/tests" -v
python -m compileall -q "$REPO/src" "$REPO/tests"
```

기존 notebook의 환경 확인 셀로 Python, NumPy, PyTorch, LightGBM, CUDA와 Git SHA를
먼저 기록한다. PyTorch를 자동 설치하거나 upgrade하지 않는다. 필요한 declared
dependency가 없거나 호환되지 않으면 project-data command 전에 중단한다.

공통 경로는 다음처럼 둔다. Dataset slug는 고정하지 않는다.

```bash
export CONFIG="$REPO/configs/phase5a_validation_robustness.json"
export INPUT_ROOT=/kaggle/input
export WORK_ROOT=/kaggle/working/phase5a-temp
export RUNS_ROOT=/kaggle/working/phase5a
mkdir -p "$WORK_ROOT" "$RUNS_ROOT"
```

## 5. 실행 순서

### 5.1 integrity-only

```bash
mkdir -p "$RUNS_ROOT/integrity_only_seed42"
python -m cell_msca.phase5a --config "$CONFIG" integrity \
  --input-root "$INPUT_ROOT" \
  --working-root "$WORK_ROOT" \
  --repository-root "$REPO" \
  --train-seed 42 \
  --kaggle | tee "$RUNS_ROOT/integrity_only_seed42/integrity_summary.json"
```

성공 출력에는 `npz_count=36`, 네 provenance hash, split별 cell count,
`test_assignment_integrity_verified=true`, `test_sample_arrays_materialized=false`가
포함되어야 한다.

### 5.2 synthetic smoke

Phase 4.3의 `configs/kaggle_synthetic_smoke.json`과 runner를 재사용해 네 variant의
CPU smoke를 수행한다. 이는 생성 데이터만 사용하며 `v1_legacy`를 열지 않는다.

```bash
for variant in token_no_attention forward reverse bidirectional; do
  python -m cell_msca.kaggle_runner \
    --variant "$variant" \
    --train-seed 3407 \
    --config "$REPO/configs/kaggle_synthetic_smoke.json" \
    --data-path generated \
    --output-path "$RUNS_ROOT/synthetic_smoke" \
    --device cpu \
    --expected-git-sha "$PHASE5A_SHA" \
    --repository-root "$REPO" \
    --kaggle
done
```

### 5.3 raw/log1p convergence diagnostic

두 command는 서로 다른 output root를 사용한다. `--allow-full-validation`은
integrity와 smoke 확인 후 사용자가 validation-only 실행을 명시적으로 여는
flag다.

```bash
python -m cell_msca.phase5a --config "$CONFIG" lightgbm-convergence \
  --input-root "$INPUT_ROOT" --working-root "$WORK_ROOT" \
  --repository-root "$REPO" --kaggle \
  --model lightgbm_raw \
  --output-root "$RUNS_ROOT/convergence_raw_seed42" \
  --expected-git-sha "$PHASE5A_SHA" --allow-full-validation

python -m cell_msca.phase5a --config "$CONFIG" lightgbm-convergence \
  --input-root "$INPUT_ROOT" --working-root "$WORK_ROOT" \
  --repository-root "$REPO" --kaggle \
  --model lightgbm_log1p \
  --output-root "$RUNS_ROOT/convergence_log1p_seed42" \
  --expected-git-sha "$PHASE5A_SHA" --allow-full-validation
```

### 5.4 LightGBM 선택 동결

위 두 output root 아래의 실제 experiment directory를 각각 지정한다.

```bash
python -m cell_msca.phase5a --config "$CONFIG" freeze-lightgbm-selection \
  --raw-run-dir <RAW_EXPERIMENT_DIRECTORY> \
  --log1p-run-dir <LOG1P_EXPERIMENT_DIRECTORY> \
  --output "$RUNS_ROOT/selection/FROZEN_PHASE5A_LIGHTGBM_SELECTION.json" \
  --expected-git-sha "$PHASE5A_SHA"
```

이 파일이 생성되기 전에는 selected LightGBM seed 43/44 실행이 거부된다. 파일의
model, parameter, provenance/config/Git hash가 다르면 역시 중단한다. 파일을 다시
읽을 때 candidate row schema와 finite validation original-unit MAE를 검사하고, 같은
tie-break 순서로 승자를 재계산하므로 저장된 `selected_model_name` 변조도 거부한다.

### 5.5 seed 43/44 반복

CPU 담당은 선택된 LightGBM만 실행한다.

```bash
for seed in 43 44; do
  python -m cell_msca.phase5a --config "$CONFIG" lightgbm-repeat \
    --input-root "$INPUT_ROOT" --working-root "$WORK_ROOT" \
    --repository-root "$REPO" --kaggle \
    --seed "$seed" \
    --selection "$RUNS_ROOT/selection/FROZEN_PHASE5A_LIGHTGBM_SELECTION.json" \
    --output-root "$RUNS_ROOT/selected_lightgbm_seed${seed}" \
    --expected-git-sha "$PHASE5A_SHA" --allow-full-validation
done
```

GPU 담당은 두 신경망 후보만 실행한다.

```bash
for seed in 43 44; do
  for variant in bidirectional token_no_attention; do
    python -m cell_msca.phase5a --config "$CONFIG" cell-repeat \
      --input-root "$INPUT_ROOT" --working-root "$WORK_ROOT" \
      --repository-root "$REPO" --kaggle \
      --seed "$seed" --variant "$variant" --device cuda \
      --output-root "$RUNS_ROOT/${variant}_seed${seed}" \
      --expected-git-sha "$PHASE5A_SHA" --allow-full-validation
  done
done
```

### 5.6 집계와 paired cell-cluster bootstrap

`<SEED42_PACKAGE>`는 private Dataset에 첨부된 기존 result package root다. 이 단계는
prediction CSV만 읽으며 test dataset을 열지 않는다.

```bash
python -m cell_msca.phase5a --config "$CONFIG" aggregate \
  --seed42-package-root <SEED42_PACKAGE> \
  --phase5a-results-root "$RUNS_ROOT" \
  --selection "$RUNS_ROOT/selection/FROZEN_PHASE5A_LIGHTGBM_SELECTION.json" \
  --repository-root "$REPO" \
  --output-dir "$RUNS_ROOT/aggregation" \
  --expected-new-git-sha "$PHASE5A_SHA"
```

집계기는 모든 prediction CSV에서 metric과 Spearman을 다시 계산한다. 세 seed의
cell_id, row order, true original/log target이 완전히 같아야 한다. 모델별 MAE,
RMSE, R², bias, Spearman의 mean과 sample standard deviation(`ddof=1`)을 기록한다.

bootstrap은 validation 고유 cell_id를 복원추출하고 선택된 각 cell의 36개월을 한
cluster로 취급한다. 비교 방향은
`MAE(bidirectional) - MAE(comparator)`이며 comparator는 token-no-attention과
선택된 LightGBM이다. 10,000회, seed 3407, 95% percentile CI를 사용한다. 이는
세 seed 평균 prediction에 조건부인 cell sampling 불확실성으로, training-seed
불확실성까지 bootstrap하는 절차는 아니다.

## 6. 결과 및 실패 조건

Phase 5A result와 selection/aggregation 파일은 모두 validation-only다. 모든 run은
고유 output root를 사용하며 기존 path가 존재하면 덮어쓰지 않고 중단한다. 표준
Cell-MSCA artifact에는 resolved config, manifest, environment, validation metrics,
validation predictions, selected checkpoint, execution log가 포함된다. LightGBM
diagnostic에는 prediction, metrics, environment, manifest가 포함된다.

다음 중 하나면 후속 실행을 진행하지 않는다.

- archive/expanded input 후보가 0개 또는 둘 이상;
- 36개 월, file hash 또는 canonical content hash 불일치;
- data/split/split-config/preprocessing/config/Git hash 불일치;
- 기존 seed 42 MAE 또는 config hash 불일치;
- prediction row/cell_id/true target 정렬 불일치;
- manifest의 `test_subset_materialized`가 `false`가 아님;
- output directory가 이미 존재함;
- LightGBM selection 파일이 없거나 변경됨.

`test` 실행 subcommand는 제공하지 않는다.
