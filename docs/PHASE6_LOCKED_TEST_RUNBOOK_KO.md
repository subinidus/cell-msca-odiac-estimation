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

실행 가능한 gate는 실제 `BaselineDataProtocol` instance 없이 만들 수 없다. gate에는 해당 protocol, 내부 dataset 및 persistent split 객체의 in-process identity와 frozen provenance를 결속한다. 실행 함수는 output 디렉터리 생성이나 test materialization 전에 `protocol is gate.bound_protocol` 및 dataset/split 객체 identity, runtime type, provenance fingerprint를 다시 검사한다. 같은 hash를 가진 별도 protocol instance도 기존 gate를 사용할 수 없다. 이 객체 결속은 직렬화 가능한 허가 토큰이 아니며 gate를 파일에 저장해 다른 process에서 재사용할 수 있다고 주장하지 않는다.

gate에는 config 파일 SHA-256과 canonical SHA-256, protocol fingerprint, 두 validation package SHA-256, data/split/split-config/preprocessing SHA-256, 실행 source Git SHA, 9개 model/seed 목록 및 deterministic `execution_id`도 고정된다. deterministic ID의 provenance 기반 구성은 유지된다.

`execution_id`는 working/output 경로를 포함하지 않는다. 같은 protocol/config, 두 package, data/split/split-config/preprocessing 및 source Git 조합은 항상 같은 ID를 만든다. production CLI에는 registry 경로 인자가 없다. 로컬 authoritative registry는 사용자 home의 `.cell_msca_phase6_final_test_registry`, Kaggle은 `/kaggle/working/.cell_msca_phase6_final_test_registry`로 고정된다. working/output root를 변경해도 registry는 바뀌지 않는다. atomic exclusive-create claim 상태는 `claimed → materialization_started → materialized → evaluating → finalizing → completed`이며, 오류가 나면 `failed`로 종료한다. output 경로를 바꿔도 동일 ID의 기존 claim이 있으면 기본 실행은 중단한다.

test materialization 직전과 직후에 registry를 갱신하고, 직후에는 row/cell/month 수와 `test_subset_materialized=true`를 기록한다. 각 model/seed prediction 완료 후 파일 경로와 SHA-256을 registry에 추가한다. 오류가 발생하면 완료된 model/seed와 artifact SHA, 실패한 model/seed, 예외 유형·메시지를 `phase6_failure_manifest.json`과 registry에 남긴다. `recovery_allowed=false`가 현재 동결 정책이다. `--allow-resume-failed-run`은 명시적으로 존재하지만 별도 승인 전에는 항상 중단하며, 완료 artifact를 자동 재계산하거나 다른 protocol/output 결과와 결합하지 않는다.

모든 검증을 통과하면 persistent split의 test를 실행 중 정확히 한 번 materialize한다. 이어서 2,574 cells, cell당 36개월, 총 92,664 rows인지 확인한다. 출력 경로가 이미 존재하면 빈 디렉터리여도 덮어쓰지 않는다. 중앙 registry가 없는 서로 다른 Kaggle 독립 세션까지 코드만으로 전역 차단한다고 주장하지 않는다. 세션 간 재실행 금지는 최종 execution receipt 보존과 사용자 연구 절차로 관리한다.

최종 success manifest, ZIP 및 receipt는 public 이름으로 바로 쓰지 않는다. 격리된 `.cell_msca_phase6_staging/<execution_id>` 아래에서 생성하고 SHA-256을 검증한 뒤 registry `finalizing`에 기록한다. registry가 원자적으로 `completed`가 된 후에만 public 이름으로 배치한다. completed 전환이 실패하면 public success manifest/ZIP/receipt는 존재하지 않는다. completed 후 publish가 중단된 경우 `recover_completed_final_publish()`는 completed registry와 staging SHA 및 원래 claim의 output 경로를 확인해 publish를 재개한다. 이 경로는 data materialization, prediction, training 또는 metric 계산을 호출하지 않는다. 성공·실패 recovery 시도는 registry에 기록하며, 변조되거나 다른 output을 가리키는 staging artifact를 거부한다.

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

저장 metric과 CSV 재계산 metric의 수치 비교에만 `relative tolerance=1e-12`, `absolute tolerance=1e-12`를 사용한다. package, prediction, model, manifest SHA-256은 여전히 byte-for-byte 동일해야 한다. metric audit에는 각 metric의 실제 absolute/relative delta와 적용 tolerance를 기록하며 NaN/Inf는 허용하지 않는다.

## 구현 검증 범위

Phase 6A 개발과 단위 테스트는 합성 test fixture만 사용한다. 실제 `v1_legacy` test materialization과 평가를 수행하지 않는다. Git에는 코드, 동결 config, 문서와 합성 테스트만 포함하며 NPZ, checkpoint, prediction 또는 결과 ZIP을 포함하지 않는다.
