# Phase 2.1 Review and Hardening Report

- 완료일: 2026-08-18 (Asia/Seoul)
- 범위: persistent split 및 preprocessing provenance 계약 보강
- 모델/baseline 구현: 없음
- 학습 실행: 없음

## 검토 범위

변경 전에 다음 문서와 `src/cell_msca`의 모든 Python source를 읽었다.

- `C:\Users\SAMSUNG\Downloads\IMPLEMENTATION_SPEC.md`
- `PHASE1_1_REPORT.md`
- `PHASE2_REPORT.md`
- `src/cell_msca/__init__.py`
- `src/cell_msca/data.py`
- `src/cell_msca/evaluate.py`
- `src/cell_msca/metrics.py`
- `src/cell_msca/splits.py`
- `src/cell_msca/target.py`

기존 테스트 28개도 변경 전에 실행해 모두 통과하는 것을 확인했다. 기본
`python`은 NumPy가 없는 Python 3.14 환경이었으므로, 테스트와 compileall은
작업공간에 제공된 의존성 포함 Python으로 실행했다. 패키지를 설치하지 않았다.

## 구현 결과

### 안전한 persistent split 로드

`load_persistent_split()`의 논문용 기본 경로는 이제 `metadata_json`과 현재
split `config`를 모두 요구한다. metadata가 없는 로드는 호출자가
`unsafe_allow_missing_metadata=True`를 명시한 진단 경로에서만 허용한다.
unsafe 경로도 CSV schema, dataset cell set, cell 좌표, 중복 cell, split 교집합,
월별 단일 split 배정은 계속 검증한다.

split metadata schema를 `cell_msca.split.v2`로 올렸다. 안전 로드에서는 다음을
현재 dataset/config 및 split CSV로 다시 계산해 대조한다.

- `schema_version`
- `split_mode`
- `split_seed`
- `data_sha256`
- `config`와 `config_sha256`
- `feature_metadata`와 `feature_metadata_sha256`
- `data_manifest`
- `split_sha256`
- `counts`와 split별 assignment SHA-256
- 현재 dataset/config로 재생성한 cell assignment
- buffered split의 feasibility 및 research-status metadata

이 계약 중 하나라도 다르면 해당 field 이름을 포함한 `ValueError`가 발생한다.
기존 `cell_msca.split.v1` sidecar는 감사 목적으로 유지하지만 v2 안전 로드에는
사용하지 않는다.

### Preprocessing provenance

`PreprocessingStats` schema를 `cell_msca.preprocessing.v2`로 정의하고 다음
provenance field를 추가했다.

- `schema_version`
- `data_sha256`
- `split_sha256`
- `split_name`

기존 train-cell 수, train-sample 수, train-cell SHA-256, feature-metadata
SHA-256과 통계 배열도 유지했다. `fit_train_preprocessing()`은 dataset hash를
자동 기록하고 호출자가 현재 split SHA-256을 전달하도록 한다.
`CellDataset.set_preprocessing()`은 preprocessing schema, data hash, split hash,
split 이름, feature metadata와 통계 배열 길이를 모두 검증한 뒤에만 통계를
적용한다. 따라서 같은 feature order를 사용하더라도 다른 data version 또는
다른 split에서 만든 통계는 즉시 거부된다.

Validation/test sentinel 격리 테스트는 provenance field 때문에 서로 다른
dataset hash가 기록되는 점을 분리해서 검사하도록 보완했다. 실제 imputation,
mean, standard deviation 및 train-cell provenance는 sentinel 크기와 무관하게
동일함을 계속 검증한다.

### 실제 비율과 buffered 거리

v2 metadata의 `counts`에 `actual_cell_ratios`, `dropped_cell_ratio`,
`dropped_sample_ratio`를 추가했다.

실제 cell-fixed split:

| Split | Cell ratio |
|---|---:|
| train | 0.6999475432767966 |
| validation | 0.15002622836160168 |
| test | 0.15002622836160168 |
| dropped | 0.0 |

실제 buffered-block split:

| Split | Cell ratio |
|---|---:|
| train | 0.23220842804686134 |
| validation | 0.14326513959316897 |
| test | 0.10887684327096812 |
| dropped | 0.5156495890890016 |

buffered metadata에는 다음 최소 Chebyshev 거리를 각각 기록한다.

| Pair | Minimum distance (cells) |
|---|---:|
| train-validation | 17 |
| train-test | 17 |
| validation-test | 1 |

`block_size=24`와 `buffer_cells=16`은 기술적으로 실행 가능한 개발 설정이지만
아직 연구적으로 확정된 protocol choice가 아니다. 이 상태를 config docstring,
buffered v2 metadata의 `research_status`, 이 보고서에 명시했다.

## Persistent artifact 보존 및 추가

기존 split assignment와 v1 metadata는 덮어쓰지 않았다. 실제 36개월
`v1_legacy` dataset과 현재 config로 assignment를 재생성해 기존 CSV와 동일한지
검증한 뒤 새 sidecar만 추가했다.

| Artifact | SHA-256 | 상태 |
|---|---|---|
| `splits/cell_fixed_seed42.csv` | `8B795AB11AD741A5E161B08EE1C2D44CEFA1B21E5B6A0A724A7CA88A85CACB40` | 기존 파일 불변 |
| `splits/buffered_block_seed42.csv` | `6604A257327A3505A3E48E83B1DF6DD8595485B9E091A2B4D89D8B9B3CC54EBA` | 기존 파일 불변 |
| `splits/cell_fixed_seed42.metadata.json` | `91B6A59B4C3EED1735C978E95F140EB118605F5290EA85DC0DB513CE729EF87C` | 기존 v1 불변 |
| `splits/buffered_block_seed42.metadata.json` | `688D8E7F6EA88A6F2EAB6FA6E6E88F562158444017E4D0311FDDFD31FA53E2D2` | 기존 v1 불변 |
| `splits/cell_fixed_seed42.metadata.v2.json` | `949E21F0C282BFA9EE5DF8F1F8D8FD8886713DABEF3DD3948FDDC872847018AE` | 신규 |
| `splits/buffered_block_seed42.metadata.v2.json` | `785D8C370BBBB32E7ECD36FB191B388BE980A29B8EDC2C15233D63ACB889436B` | 신규 |

두 v2 sidecar를 사용한 실제 cell-fixed 및 buffered-block 안전 재로드도 각각
통과했다. 실제 data SHA-256은 기존과 동일한
`c3a1889fd863117c3faa9125094267974dbacfe6fd4ca6c157dead2d7c874b62`다.

## 테스트 및 compileall

실행 결과:

- 전체 테스트: **34개 실행, 34개 통과**
- 기존 테스트: **28개 유지, 전부 통과**
- 신규 회귀 테스트: **6개, 전부 통과**
- `python -m compileall -q src tests`: **통과**
- 99자를 초과하는 `src/cell_msca` 및 test source line: **0개**

신규 테스트는 다음을 검증한다.

1. metadata 없는 안전 로드 실패 및 명시적 unsafe 경로 허용
2. 변경된 data SHA-256에서 안전 로드 실패
3. 변경된 feature metadata에서 안전 로드 실패
4. schema, mode, seed, config, manifest, counts 등 metadata field별 변조 실패
5. 다른 split SHA-256 및 split 이름의 preprocessing 통계 적용 실패
6. 다른 data version의 preprocessing 통계 적용 실패

기존 buffered 테스트에는 actual dropped ratio, 세 split 쌍의 Chebyshev 거리,
development-only research status 검증을 추가했다. 기존 train-only sentinel
테스트에는 새 preprocessing provenance 검증을 추가했다.

## 변경 파일

### Source

- `src/cell_msca/__init__.py`
- `src/cell_msca/data.py`
- `src/cell_msca/splits.py`

### Tests

- `tests/test_splits.py`

### New artifacts and report

- `splits/cell_fixed_seed42.metadata.v2.json`
- `splits/buffered_block_seed42.metadata.v2.json`
- `PHASE2_1_REPORT.md`

기존 flat `src/*.py` 11개와 그 안의 legacy dataset/model/baseline/train source,
기존 Phase 보고서 및 기존 split/result artifact는 수정하지 않았다.

## Flat source SHA-256 불변 검증

현재 flat `src/*.py` 11개를 공급된
`C:\Users\SAMSUNG\Downloads\msca-carbon-main.zip`의 대응 entry와 비교했다.
11개 모두 일치했고 mismatch는 0개였다.

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

- split assignment 생성 알고리즘과 기존 CSV는 변경하지 않았다.
- model 및 baseline을 구현하지 않았다.
- train 코드를 수정하거나 실제 학습을 실행하지 않았다.
- 기존 split, 결과, checkpoint를 수정·삭제·덮어쓰지 않았다.
- 기존 v1 metadata는 v2 안전 계약에 필요한 actual ratios와 provenance 검증
  계약이 없으므로 논문용 로드에는 신규 `.metadata.v2.json`을 사용해야 한다.
- `block_size=24`, `buffer_cells=16`의 연구 protocol 확정은 후속 실험 설계
  단계의 미해결 사항이다.
- `v1_legacy`의 target unit 및 과거 전처리 한계는 `PHASE2_REPORT.md`에 기록된
  상태와 동일하며 이번 범위에서 변경하지 않았다.
