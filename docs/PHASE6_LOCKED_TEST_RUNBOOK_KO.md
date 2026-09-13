# Phase 6A locked final-test 실행 안내

이 실행기는 Phase 5A validation에서 동결된 세 모델과 train seed 42, 43, 44만 평가한다. 학습, early stopping, checkpoint 재선택, hyperparameter 변경 및 test 결과 기반 모델 선택 경로는 제공하지 않는다. 대상은 `v1_legacy ODIAC-derived zonal mean`이며, 직접 측정 배출량이나 target-cell 총량으로 해석하지 않는다.

## 동결 입력

- 소스 기준: Git commit `35696c7e8c251e47e12c095f4e9e648a43f97a42`
- Phase 5A 결과: `cell-msca-phase5a-full-validation-35696c7e8c25.zip`
  - SHA-256: `b4fb423e9d545a7fab654e6049416898bdc31ea3a64c6c7e414ad727ec74e7a3`
- seed-42 결과: `cell-msca-v1-legacy-validation-034712a.zip`
  - SHA-256: `f40aa8db3cff1a56933a69abe9397f43e8df9daa642319848c928cea4ed98555`
- 데이터와 persistent split의 정확한 hash 및 9개 model artifact identity는 `configs/phase6_locked_test_protocol.json`에 고정되어 있다.

9개 artifact의 배치는 다음과 같다.

| 모델 | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| `lightgbm_raw` | Phase 5A ZIP | Phase 5A ZIP | Phase 5A ZIP |
| `cell_msca_token_no_attention` | seed-42 ZIP | Phase 5A ZIP | Phase 5A ZIP |
| `cell_msca_bidirectional` | seed-42 ZIP | Phase 5A ZIP | Phase 5A ZIP |

## test gate

`--allow-final-test`가 없으면 파일 검증 전에 종료한다. 인자가 있더라도 다음 검증이 모두 끝나기 전에는 test authorization을 만들지 않는다.

1. config schema, 모델/seed 집합, metric 및 bootstrap 계약
2. 두 ZIP의 파일명과 전체 SHA-256
3. Phase 5A progress의 모든 stage 완료와 failed stage 0
4. aggregation manifest의 9개 validation input identity
5. LightGBM validation 승자와 동결 parameter
6. 각 run manifest, validation prediction, checkpoint/fitted-model SHA-256
7. validation-only 상태와 닫힌 test gate
8. LightGBM text model 및 PyTorch `weights_only=True` checkpoint의 안전한 load
9. project data/split/split-config/preprocessing hash
10. output 경로가 존재하지 않음

seed-42 neural manifest는 이전 schema라 `test_evaluation_performed` 필드가 없다. 이 예외는 정확한 seed-42 ZIP hash와 두 manifest hash에만 한정되며, `test_subset_materialized=false`, Phase 5A aggregation의 closed-gate audit 및 validation prediction identity를 함께 요구한다. 다른 누락은 허용하지 않는다.

모든 검증을 통과하면 persistent split의 test를 실행 중 정확히 한 번 materialize한다. 이어서 2,574 cells, cell당 36개월, 총 92,664 rows인지 확인한다. 출력 경로가 이미 존재하면 빈 디렉터리여도 덮어쓰지 않는다. 서로 다른 작업 공간이나 세션에서 사용자가 출력을 삭제하고 다시 실행하는 행위까지 소프트웨어가 전역적으로 막을 수는 없으므로, 최종 ZIP과 receipt를 보존하고 동일 protocol을 다시 실행하지 않는 운영 통제가 필요하다.

## 실행 명령

아래 명령은 실제 final test를 연다. 리뷰 승인과 실행 책임자의 명시적 결정 전에는 실행하지 않는다.

```powershell
python -m cell_msca.phase6 `
  --config configs/phase6_locked_test_protocol.json `
  --input-root <36-NPZ 또는 2022_2024_npz.zip을 포함한 입력 루트> `
  --working-root <안전한 임시 작업 루트> `
  --repository-root . `
  --phase5a-zip <cell-msca-phase5a-full-validation-35696c7e8c25.zip> `
  --seed42-zip <cell-msca-v1-legacy-validation-034712a.zip> `
  --output-dir <존재하지 않는 고유 출력 경로> `
  --device cpu `
  --allow-final-test
```

CUDA에서 neural inference를 수행하려면 `--device cuda`를 명시한다. `auto`도 허용하지만 논문 재현 실행에서는 실제 device를 사전에 고정하는 편이 명확하다. 다중 GPU 분산 실행은 하지 않는다.

## 결과 계약

각 model/seed 디렉터리는 projected `test_predictions.csv`, `test_metrics.json`, 음수 projection 행만 담은 diagnostic CSV 및 run manifest를 가진다. `pred_original`에는 validation과 동일한 `nonnegative_max_zero_v1` 정책이 적용되고, 보정 전 음수 개수·비율·최솟값·적용 개수를 별도로 기록한다. true target은 변경하지 않는다.

전체 결과에는 다음이 포함된다.

- `PHASE6_LOCKED_TEST_PROTOCOL.json`
- 9개 run의 prediction, metric, diagnostic 및 manifest
- `test_seed_metric_summary.json` (`ddof=1`)
- `paired_test_cell_cluster_bootstrap.json` (10,000회, seed 3407)
- `phase6_test_manifest.json`
- `final_test_gate_audit.json`
- `execution.log`
- 출력 디렉터리의 sibling final ZIP 및 SHA-256 receipt

Primary metric은 original-unit MAE, RMSE, R²이다. Bias, Spearman과 log-unit MAE/RMSE/R²는 secondary다. 모든 metric은 저장된 test prediction CSV에서 다시 계산한다. Bootstrap은 unique test cell을 복원추출하고 각 선택 cell의 36개월 행을 함께 사용한다. 비교값은 세 frozen seed prediction의 rowwise mean이며, 이 bootstrap은 training-seed uncertainty를 포함하지 않는다.

## 구현 검증 범위

Phase 6A 개발과 단위 테스트는 합성 test fixture만 사용한다. 실제 `v1_legacy` test materialization과 평가를 수행하지 않는다. Git에는 코드, 동결 config, 문서와 합성 테스트만 포함하며 NPZ, checkpoint, prediction 또는 결과 ZIP을 포함하지 않는다.
