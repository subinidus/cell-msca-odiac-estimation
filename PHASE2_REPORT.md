# Phase 2 Implementation Report

- 완료일: 2026-08-22 (Asia/Seoul)
- 범위: single-cell dataset, persistent splits, train-only preprocessing
- 전체 데이터 학습: 실행하지 않음

## 구현 결과

### CellDataset

`src/cell_msca/data.py`에 16x16 patch를 만들지 않는 `CellDataset`을
구현했다. 모델-facing sample은 다음 여섯 필드만 반환한다.

- `stream_a`: shape `[3]`
- `stream_b`: shape `[4]`
- `target_original`
- `target_log`
- `cell_id`
- `month_id`

주변 셀, 좌표, row/column index, mask, hotspot label, `label_cls`는 sample에
포함하지 않는다. row/column, data version, target unit, feature order 등 감사용
정보는 `sample_metadata()`의 별도 경로에서만 조회한다.

NPZ의 Stream A/B feature 이름과 순서를 다음 계약과 정확히 비교한다.

- Stream A: `no2_mean`, `so2_mean`, `co_mean`
- Stream B: `nightlight_mean`, `urban_fraction`, `power_plant_count`,
  `fossil_capacity_mw`

신규 key인 `stream_a_feature_names`/`stream_b_feature_names`와 legacy key인
`stream_a_features`/`stream_b_features`를 모두 읽지만, 값의 이름과 순서는
동일하게 엄격히 검증한다. Legacy NPZ에 없는 `data_version`과 `target_unit`은
호출자가 두 값을 함께 명시해야만 로드할 수 있다.

모든 월의 mask가 동일한지도 검사한다. 따라서 유효한 한 cell은 모든 월에
존재하며 하나의 persistent split 배정을 공유한다.

### Train-only preprocessing

`fit_train_preprocessing()`은 명시적으로 전달된 train `cell_id`의 모든 월만
사용해 다음 통계를 계산한다.

1. feature별 관측값 평균을 이용한 결측치 대체값
2. 결측치를 대체한 train 표본의 평균과 표준편차
3. `(x - train_mean) / (train_std + epsilon)` 정규화

통계에는 train cell 수, train sample 수, train-cell SHA-256, feature metadata
SHA-256이 포함된다. JSON-compatible dictionary 변환과 복원을 지원한다.
Validation/test sentinel을 바꾼 두 합성 dataset의 통계가 정확히 동일한지
검증했다.

실제 `v1_legacy` 36개월 자료에서도 train cell만 사용해 통계를 계산했다.

- train cells: 12,009
- train samples: 432,324
- preprocessing SHA-256:
  `e22e40fa151ed0a0dd1dd07d8273e9d2e0a646f7b2fa2fa73c941bbf37950dcf`

이 통계는 검증 목적으로만 계산했고 별도 결과 파일이나 체크포인트로
저장하지 않았다.

### Persistent cell-fixed split

`src/cell_msca/splits.py`에 split 생성, CSV 저장, metadata JSON 저장, 재로드,
checksum 검증, cell 교집합 검증, sample subset 복원 경로를 구현했다.

`SeedConfig`는 `split_seed`와 `train_seed`를 별도 필드로 보유한다.
`CellFixedSplitConfig.from_seeds()`는 `split_seed`만 읽고 `train_seed`는 split
config와 checksum에서 제외한다. 서로 다른 train seed 42와 3407에서 동일한
split checksum이 생성되는 것을 테스트했다.

실제 `v1_legacy` mask로 다음 영구 파일을 생성했다.

- `splits/cell_fixed_seed42.csv`
- `splits/cell_fixed_seed42.metadata.json`

결과:

| Split | Cells | Samples | Assignment SHA-256 |
|---|---:|---:|---|
| train | 12,009 | 432,324 | `9b9ecd3293a23725324eb6f2de4c200372232f267eec902eed573971e98c064e` |
| validation | 2,574 | 92,664 | `efe9582951828be92da859f89924f357997e40cf0c533bc6ac9c2f856f9fe25a` |
| test | 2,574 | 92,664 | `517e93beb165dd1d26d6f8fbfb361f096840405a6eeb998ca3da2651ac32a064` |

- cell intersections: 모두 0
- split CSV SHA-256:
  `8b795ab11ad741a5e161b08ee1c2d44cefa1b21e5b6a0a724a7ca88a85cacb40`
- data SHA-256:
  `c3a1889fd863117c3faa9125094267974dbacfe6fd4ca6c157dead2d7c874b62`
- split config SHA-256:
  `b790de004c019af6842da99723c7d608d6a5bfa454d70b3053f26eca4287cb8b`

Metadata JSON에는 전체 cell/sample 수, split별 cell/sample 수, split별 assignment
SHA-256, split/data/config SHA-256, feature metadata 및 36개 NPZ file manifest가
기록된다. 저장 함수는 기존 split 파일을 덮어쓰지 않는다.

### Buffered spatial block split

Cell-fixed 주 경로와 분리된 다음 API를 구현했다.

- `assess_buffered_block_split()`
- `create_persistent_buffered_block_split()`

실제 `v1_legacy` mask에서 `block_size=24`, `buffer_cells=16`, `split_seed=42`로
feasibility를 확인했고 non-empty train/validation/test와 최소 Chebyshev 거리
17을 확보했다. 따라서 별도 robustness split을 생성했다.

- `splits/buffered_block_seed42.csv`
- `splits/buffered_block_seed42.metadata.json`

| Split | Cells | Samples |
|---|---:|---:|
| train | 3,984 | 143,424 |
| validation | 2,458 | 88,488 |
| test | 1,868 | 67,248 |
| dropped buffer | 8,847 | 318,492 |

- split CSV SHA-256:
  `6604a257327a3505a3e48e83b1df6dd8595485b9e091a2b4d89d8b9b3cc54eba`
- config SHA-256:
  `3e6e7e52d0c7add6f1217485d174778c2b07a70679bfc88ed0585f5a227b5ca7`
- minimum train-to-heldout Chebyshev distance: 17 cells

기술적으로는 feasible이지만 8,847개 셀, 약 51.6%가 buffer로 제외된다. 이
robustness split의 통계적 효율성과 최종 실험 포함 여부는 이후 실험 프로토콜에서
명시적으로 검토해야 한다.

## 변경 및 생성 파일

### Source

- `src/cell_msca/data.py` (신규)
- `src/cell_msca/splits.py` (신규)
- `src/cell_msca/__init__.py` (Phase 2 public API export 추가)

### Tests

- `tests/synthetic_npz.py` (신규)
- `tests/test_cell_dataset.py` (신규)
- `tests/test_splits.py` (신규)

### Persistent artifacts

- `splits/cell_fixed_seed42.csv` (신규)
- `splits/cell_fixed_seed42.metadata.json` (신규)
- `splits/buffered_block_seed42.csv` (신규)
- `splits/buffered_block_seed42.metadata.json` (신규)
- `PHASE2_REPORT.md` (신규)

기존 flat `src/*.py`, 기존 Phase 1 모듈의 구현, dataset/model/train/baseline
legacy 코드는 수정하지 않았다.

## 테스트 및 compileall

실행 명령:

```powershell
python -m unittest discover -s tests -t . -v
python -m compileall -q src\cell_msca tests
```

결과:

- 전체 단위·통합 테스트: **28개 실행, 28개 통과**
- 기존 Phase 1/1.1 테스트: **17개 유지, 전부 통과**
- 신규 Phase 2 테스트: **11개, 전부 통과**
- `compileall`: **통과**
- 99자를 초과하는 새 source/test line: 0
- Ruff: 실행 환경에 executable이 없어 미실행

검증한 Phase 2 조건:

- `[3]`/`[4]` single-cell feature shape
- model-facing sample에서 공간·좌표·hotspot/classification 필드 제외
- feature 이름 및 순서 불일치 즉시 거부
- 월별 mask 불일치 거부
- persistent split 저장·재로드 및 checksum 검증
- train/validation/test cell 교집합 0
- 동일 cell의 모든 월이 하나의 split에만 존재
- train seed 변경 시 split checksum 불변
- validation/test sentinel이 imputation/normalization 통계에 영향 없음
- split별 cell/sample/assignment checksum metadata 기록
- buffered-block feasibility 성공/실패 경로와 별도 persistence 경로

## Flat source 불변 및 제약 확인

공급된 `msca-carbon-main.zip`과 현재 flat `src/*.py` 11개를 SHA-256으로 다시
비교했다.

- 비교 파일: 11개
- hash mismatch: 0개
- 새 model/train/baseline 모듈: 0개
- `.pth`, `.pt`, `.ckpt` 생성: 0개
- 전체 데이터 학습 실행: 없음
- 기존 결과 또는 체크포인트 덮어쓰기: 없음

## 남은 사항

- 실제 `v1_legacy` NPZ는 전처리 단계에서 월 전체 cell을 사용한 결측치 대체가
  이미 반영되어 있다. 현재 train-only imputation은 이 과거 제한을 되돌릴 수
  없으며, 실제 missing-value leakage 검증은 합성 자료에서 수행했다.
- 실제 `v2_corrected`가 제공되면 원시 결측치를 보존한 상태로 동일 sentinel
  검증을 다시 실행해야 한다.
- `v1_legacy` split metadata의 target unit은 명세에 따라
  `ODIAC native value; exact target-cell interpretation unverified`로 기록했다.
  물리 단위가 확인되기 전에는 corrected physical target으로 해석하면 안 된다.
- 생성된 split은 metadata의 data SHA-256과 일치하는 자료에만 재사용해야 한다.
- 모델, baseline 및 학습 코드는 Phase 2 범위에서 구현하거나 실행하지 않았다.
