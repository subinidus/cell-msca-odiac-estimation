# v1_legacy frozen validation 실행 안내서

## 1. 범위와 고정된 연구 계약

이 패키지는 `2022_2024_npz.zip`과 기존
`cell_fixed_seed42` split을 사용해 train/validation만 실행한다. test assignment의
무결성은 split metadata에서 확인하지만 test sample array를 만들거나 metric을 계산하지
않는다. 전체 validation 학습은 기본값이 꺼져 있으며 명시적 실행 인자가 필요하다.

`label_reg`는 legacy 전처리가 프로젝트 grid와 교차한다고 선택한 ODIAC source-pixel
월별 값의 산술평균으로 취급한다. target-cell 총량, 직접 측정 배출량, 질량보존 target으로
해석하지 않는다. 물리 단위는 아직 확인되지 않았다. archive의 feature에는 persistent
split 생성 전 월별 전체 grid 통계로 수행된 upstream 결측치 대체가 포함되어 있으며,
현재 train-only preprocessing은 이를 되돌릴 수 없다.

고정값은 다음과 같다.

- data version: `v1_legacy`
- split seed: `42`
- first comparison train seed: `42`
- selection: validation original-unit MAE
- headline: original-unit MAE, RMSE, R²
- secondary: log-space metrics, Spearman
- log inverse: checkpoint 확정 뒤 validation에서 median/Duan 비교, Duan residual은 train만 사용
- materialization: train, validation만 허용
- overwrite: 금지

## 2. 입력과 무결성 기준

Kaggle private Dataset에는 다음 파일 하나만 올린다.

- `2022_2024_npz.zip`

Dataset slug나 mount 경로는 고정하지 않는다. 검증기는 `/kaggle/input` 아래에서 같은
이름의 archive를 재귀적으로 찾으며 정확히 하나가 아니면 중단한다. frozen archive
manifest는 다음을 고정한다.

- archive SHA-256: `5e01b3bc58fa2adebbe24aa2b6c3d2d2b93d274adc1c2686a37c62f57a5d93ed`
- month coverage: 2022-01--2024-12, 정확히 36개
- dataset SHA-256: `c3a1889fd863117c3faa9125094267974dbacfe6fd4ca6c157dead2d7c874b62`
- split SHA-256: `8b795ab11ad741a5e161b08ee1c2d44cefa1b21e5b6a0a724a7ca88a85cacb40`
- split-config SHA-256: `b790de004c019af6842da99723c7d608d6a5bfa454d70b3053f26eca4287cb8b`
- preprocessing SHA-256: `35f760ab7a73ae561ae42562889e67cd1ac1143cdfa7ab6a5b3b25352efcc4db`

각 NPZ의 파일 크기, file SHA-256, canonical content SHA-256도
`configs/v1_legacy_archive_manifest.json`에 고정되어 있다. 압축은
`/kaggle/working`의 임시 디렉터리에만 풀고 명령 종료 시 자동 정리한다.

## 3. 소스 checkout 계약

공개 저장소와 reviewed runner 기준점은 다음과 같다.

- repository: `https://github.com/subinidus/cell-msca-odiac-estimation.git`
- reviewed base tag: `v0.4.3-kaggle-validation-runner`
- tag commit: `bb223dd82aa14ec695aff146780377748a163f10`

먼저 tag를 정확히 확인한다.

```bash
git clone https://github.com/subinidus/cell-msca-odiac-estimation.git /kaggle/working/cell-msca
cd /kaggle/working/cell-msca
git checkout --detach v0.4.3-kaggle-validation-runner
test "$(git rev-parse HEAD)" = "bb223dd82aa14ec695aff146780377748a163f10"
```

이 문서, frozen configs, Phase 3 baseline adapter는 v0.4.3 이후의 별도 검토
패키지다. 따라서 실제 validation 실행 전에는 이 PR이 병합된 뒤 담당자가 공지한
`FROZEN_PACKAGE_SHA`를 exact checkout해야 한다. v0.4.3 tag만 checkout한 상태에는
새 패키지가 없으므로 실행 준비가 완료된 것으로 간주하지 않는다.

```bash
git fetch origin --tags
git checkout --detach "$FROZEN_PACKAGE_SHA"
git merge-base --is-ancestor v0.4.3-kaggle-validation-runner HEAD
test "$(git rev-parse HEAD)" = "$FROZEN_PACKAGE_SHA"
```

attached-code 방식은 이 실험에서 사용하지 않는다. Kaggle에서 Git 접근이 안 되면
임의 소스 사본으로 우회하지 않고 실행을 중단한다. PyTorch는 Kaggle image의 호환
버전을 사용하며 자동 설치·업그레이드하지 않는다. package 자체는 dependency를
변경하지 않고 설치한다.

```bash
python -m pip install -e . --no-deps
python -c "import sys,numpy,torch; print(sys.version); print(numpy.__version__); print(torch.__version__, torch.cuda.is_available())"
python -c "import lightgbm; print(lightgbm.__version__)"
```

## 4. 지원 상태

| assignment | 장치 | 상태 | 기존 구현 경로 |
|---|---:|---|---|
| `train_mean_seed42` | CPU | 지원 | Phase 3 train-mean + v1 adapter |
| `lightgbm_raw_seed42` | CPU | 지원 | Phase 3 raw LightGBM + v1 adapter |
| `lightgbm_log1p_seed42` | CPU | 지원 | Phase 3 log1p LightGBM + v1 adapter |
| `lightgbm_tweedie_seed42` | CPU | 지원 | Phase 3 Tweedie LightGBM + v1 adapter |
| `concat_mlp_raw_huber_seed42` | GPU | 미지원 | 구현된 Concat-MLP는 log target의 `log_l1`/`log_huber`만 지원 |
| `concat_mlp_log1p_seed42` | GPU | 지원 | Phase 3 neural trainer + v1 adapter |
| `token_no_attention_seed42` | GPU | 지원 | Phase 4.3 runner |
| `cell_msca_forward_seed42` | GPU | 지원 | Phase 4.3 runner, Q=I, K/V=P |
| `cell_msca_reverse_seed42` | GPU | 지원 | Phase 4.3 runner, Q=P, K/V=I |
| `cell_msca_bidirectional_seed42` | GPU | 지원 | Phase 4.3 runner, 양방향 |

미지원 assignment는 구성값을 추정하지 않으며 실행하지 않는다. attention weight는
인과적 또는 source-attribution 근거로 해석하지 않는다.

## 5. Notebook 셀 실행 순서

기존 `notebooks/cell_msca_kaggle_validation.ipynb`를 얇은 orchestration wrapper로
재사용한다. 셀 순서는 다음과 같다.

1. Kaggle accelerator와 internet 설정을 확인한다.
2. 공개 repository를 clone하고 v0.4.3 tag SHA를 확인한다.
3. 승인된 `FROZEN_PACKAGE_SHA`를 exact checkout하고 v0.4.3가 ancestor인지 확인한다.
4. Python, NumPy, PyTorch, LightGBM, CUDA/GPU, Git SHA를 출력한다.
5. `python -m unittest discover -s tests -v`와 `python -m compileall src tests`를 실행한다.
6. 기존 synthetic runner를 CPU에서 네 variant 모두 실행한다.
7. 아래 integrity-only 명령으로 실제 archive와 split/preprocessing hash를 검증한다.
8. smoke config로 지정 assignment만 실행한다. Cell-MSCA smoke는 한 epoch다.
9. 산출물 목록과 크기를 확인하고 필요한 작은 결과만 내려받는다.
10. full validation은 별도 승인 후에만 명시적 gate 인자로 실행한다.

## 6. 실제 archive integrity-only

```bash
python -m cell_msca.v1_validation integrity \
  --input-root /kaggle/input \
  --working-root /kaggle/working/v1-legacy-integrity \
  --manifest configs/v1_legacy_archive_manifest.json \
  --repository-root /kaggle/working/cell-msca \
  --kaggle
```

성공 조건은 archive/36 NPZ의 크기와 두 종류 hash, feature order, 동일 mask,
dataset/split/split-config/preprocessing hash가 모두 일치하고 출력 JSON에
`test_sample_arrays_materialized: false`가 기록되는 것이다.

## 7. synthetic CPU smoke

각 명령은 서로 다른 빈 output root를 사용한다. `$FROZEN_PACKAGE_SHA`는 실제 exact
commit SHA로 치환한다.

```bash
for variant in token_no_attention forward reverse bidirectional; do
  python -m cell_msca.kaggle_runner \
    --variant "$variant" \
    --train-seed 3407 \
    --config configs/kaggle_synthetic_smoke.json \
    --data-path generated \
    --output-path "/kaggle/working/synthetic-$variant" \
    --device cpu \
    --expected-git-sha "$FROZEN_PACKAGE_SHA" \
    --repository-root /kaggle/working/cell-msca \
    --kaggle
done
```

네 실행 모두 finite forward/backward, checkpoint reload, prediction metric round-trip,
`status: completed`를 만족해야 한다.

## 8. 실제 자료 smoke

CPU baseline 예시는 다음과 같다. model별로 명령을 각각 실행한다.

```bash
python -m cell_msca.v1_validation baseline \
  --input-root /kaggle/input \
  --working-root /kaggle/working/v1-legacy-smoke-tmp \
  --manifest configs/v1_legacy_archive_manifest.json \
  --repository-root /kaggle/working/cell-msca \
  --kaggle \
  --model lightgbm_raw \
  --config configs/v1_legacy_baseline_smoke.json \
  --output-root /kaggle/working/v1-legacy-validation \
  --expected-git-sha "$FROZEN_PACKAGE_SHA"
```

GPU Cell-MSCA 한 epoch smoke 예시는 다음과 같다.

```bash
python -m cell_msca.v1_validation cell \
  --input-root /kaggle/input \
  --working-root /kaggle/working/v1-legacy-smoke-tmp \
  --manifest configs/v1_legacy_archive_manifest.json \
  --repository-root /kaggle/working/cell-msca \
  --kaggle \
  --variant forward \
  --config configs/v1_legacy_cell_msca_1epoch_smoke.json \
  --output-root /kaggle/working/v1-legacy-validation/cell_msca_forward_seed42 \
  --expected-git-sha "$FROZEN_PACKAGE_SHA" \
  --device cuda
```

실제 자료 smoke 완료 조건은 train/validation만 materialize하고, output overwrite가
거부되며, validation prediction과 metric의 재계산이 일치하는 것이다. GPU smoke는
선택된 checkpoint의 안전한 reload까지 포함한다.

## 9. full validation 명령

smoke가 통과하고 담당자가 실행을 승인한 뒤에만 full config와 gate 인자를 사용한다.

```bash
python -m cell_msca.v1_validation baseline \
  --input-root /kaggle/input \
  --working-root /kaggle/working/v1-legacy-full-tmp \
  --manifest configs/v1_legacy_archive_manifest.json \
  --repository-root /kaggle/working/cell-msca \
  --kaggle \
  --model lightgbm_log1p \
  --config configs/v1_legacy_baseline_validation.json \
  --output-root /kaggle/working/v1-legacy-validation \
  --expected-git-sha "$FROZEN_PACKAGE_SHA" \
  --allow-full-validation
```

```bash
python -m cell_msca.v1_validation cell \
  --input-root /kaggle/input \
  --working-root /kaggle/working/v1-legacy-full-tmp \
  --manifest configs/v1_legacy_archive_manifest.json \
  --repository-root /kaggle/working/cell-msca \
  --kaggle \
  --variant bidirectional \
  --config configs/v1_legacy_cell_msca_validation.json \
  --output-root /kaggle/working/v1-legacy-validation/cell_msca_bidirectional_seed42 \
  --expected-git-sha "$FROZEN_PACKAGE_SHA" \
  --device cuda \
  --allow-full-validation
```

## 10. 산출물과 협업 규칙

각 owner는 `configs/v1_legacy_validation_assignments.json`의 frozen assignment만
실행하고 hyperparameter를 독립적으로 바꾸지 않는다. output 경로는 assignment별로
고유하며 재사용하지 않는다. 실제 결과, prediction, checkpoint, archive는 Git에
commit하지 않는다.

Cell-MSCA 실행은 `resolved_config.json`, `run_manifest.json`, `environment.json`,
`validation_metrics.json`, `validation_predictions.csv`, `selected_checkpoint.pt`,
`execution.log`를 생성한다. Phase 3 baseline adapter는 checkpoint 공통 저장 계약이
없으므로 validation predictions/metrics, environment, run manifest만 저장하고 그
사실을 manifest에 명시한다.

실행 전 차단 조건은 archive 탐색 결과가 0개 또는 2개 이상인 경우, 어느 hash라도
불일치하는 경우, Git SHA 불일치 또는 dirty source, 기존 output 경로 존재, test 접근
요청이다.
