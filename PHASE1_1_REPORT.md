# Phase 1.1 Review and Hardening Report

- 검토일: 2026-08-12 (Asia/Seoul)
- 범위: Phase 1 evaluation core 보완
- 학습 실행: 없음

## 검토한 기준 문서와 코드

- `C:\Users\SAMSUNG\Downloads\IMPLEMENTATION_SPEC.md`
- `PHASE1_REPORT.md`
- `src/cell_msca/__init__.py`
- `src/cell_msca/target.py`
- `src/cell_msca/metrics.py`
- `src/cell_msca/evaluate.py`
- 기존 테스트 12개

## 구현 내용

1. `metrics_by_space()`가 original-unit 배열 쌍과 log-space 배열 쌍을 각각
   검증한 뒤 두 공간의 표본 수를 비교하도록 보완했다. 표본 수가 다르면
   두 크기를 포함한 `ValueError`를 발생시킨다.
2. `inverse_target()`이 median 및 Duan-smearing inverse 계산 후 결과 전체의
   유한성을 확인하도록 보완했다. overflow 등으로 NaN 또는 inf가 생성되면
   inverse mode와 비유한 값 개수를 포함한 `ValueError`를 발생시킨다.
3. 공개 함수 `read_prediction_csv()`를 추가했다. `cell_id`는 숫자로 변환하지
   않고 CSV의 문자열 표현 그대로 보존하며, 네 prediction/target 열은
   `float64` 배열로 복원한다.
4. `calculate_prediction_bootstrap_from_csv()`를 추가했다. 저장된 prediction
   CSV의 값과 `cell_id`만 사용해 original-unit 및 log-space cell-cluster
   bootstrap을 다시 계산한다.
5. bootstrap의 cell draw와 전체 행 복원을 `_draw_complete_cell_indices()`로
   분리했다. 고정 seed에서 결과가 결정론적이며, 복원추출된 각 cell의 모든
   행이 하나의 완전한 묶음으로 포함되는지 테스트한다.
6. Spearman은 이번 구현의 headline metric에 추가하지 않았다. Phase 3의
   baseline 공통 evaluator에서 secondary metric으로 추가하고, original-unit
   headline MAE/RMSE/R2/Bias와 별도로 명명하는 작업으로 남긴다.

## 변경 파일

- `src/cell_msca/__init__.py`
- `src/cell_msca/target.py`
- `src/cell_msca/metrics.py`
- `src/cell_msca/evaluate.py`
- `tests/test_target.py`
- `tests/test_metrics.py`
- `tests/test_prediction_roundtrip.py`
- `PHASE1_1_REPORT.md` (신규)

기존 flat `src/*.py`, dataset, split, model, train 코드는 수정하지 않았다.
기존 `PHASE1_REPORT.md`도 변경하지 않았다.

## 테스트 결과

실행 명령:

```powershell
python -m unittest discover -s tests -t . -v
```

결과: **17개 실행, 17개 통과**.

- 기존 Phase 1 테스트: 12개 유지 및 전부 통과
- 신규 보완 테스트: 5개 추가 및 전부 통과
  - original/log 공간 간 표본 수 불일치 거부
  - inverse 결과의 비유한 값 거부
  - prediction CSV `cell_id` 보존
  - prediction CSV 기반 bootstrap의 직접 계산 결과 재현
  - 고정 seed에서 complete-cell 묶음 복원추출 검증

구문 컴파일:

```powershell
python -m compileall -q src\cell_msca tests
```

결과: **통과**.

## Flat source SHA-256 불변 검증

현재 파일을 수정 전 기준값 및 공급된 `msca-carbon-main.zip`의 대응 entry와
각각 비교했다. 11개 모두 일치했고 mismatch는 0개였다.

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

## 제약 준수 및 남은 사항

- dataset, split, model, train 코드를 수정하지 않았다.
- 학습 명령을 실행하지 않았다.
- 결과, 체크포인트, `.pth`, `.pt`, `.ckpt` 파일을 생성·삭제·덮어쓰지 않았다.
- median과 Duan-smearing 중 최종 inverse mode 선택은 validation 단계에서
  확정하고 test split 평가 전에 기록해야 한다.
- ODIAC 물리 단위와 target transform의 `scale`은 별도 검증이 필요하다.
- Spearman은 Phase 3에서 secondary metric으로 추가할 예정이다.
