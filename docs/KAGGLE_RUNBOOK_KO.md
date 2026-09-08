# Cell-MSCA Kaggle validation 실행 지침

## 1. 목적과 범위

이 지침은 검토된 `cell_msca` package를 Kaggle에서 재현 가능하게 실행하기
위한 engineering/validation 전용 절차다. notebook은 환경 확인과 명령 실행만
담당한다. 모델, split, preprocessing, inverse 선택, metric, checkpoint 로직은
`src/cell_msca`의 기존 구현을 사용한다.

다음 작업은 이 경로에서 허용되지 않는다.

- test subset materialization 또는 평가;
- `v1_legacy`를 사용한 전체 학습을 synthetic smoke로 가장하는 행위;
- 개인별 hyperparameter 변경;
- `/kaggle/input`에 파일 쓰기;
- 기존 output directory 덮어쓰기;
- notebook 또는 config에 token, password, credential, private URL 삽입;
- PyTorch 자동 설치 또는 upgrade.

## 2. 사전 조건

- 검토 기준 태그: `v0.4.0-cell-msca-architecture`;
- Python 3.10 이상;
- PyTorch 2.1 이상;
- NumPy 1.24 이상;
- Cell-MSCA 실행에는 LightGBM이 필요하지 않지만 환경 manifest에는 설치 여부와
  버전을 기록한다;
- Kaggle에서는 Internet 설정과 관계없이 `/kaggle/input`을 read-only로 취급하고
  `/kaggle/working` 아래만 output으로 사용한다.

runner는 tracked 및 untracked 파일이 없는 정확한 Git checkout을 요구한다. private
code dataset은 `GIT_COMMIT_SHA.txt`, `SOURCE_TREE_MANIFEST.json`, 그리고 coordinator가
별도 전달한 manifest SHA-256을 함께 제공해야 한다. SHA 선언만으로 source를 검증된
상태로 취급하지 않는다. manifest는 `src/cell_msca`, 전체 `configs`, Kaggle notebook,
`pyproject.toml`, `requirements.txt`의 repository-relative path와 SHA-256을 기록한다.

## 3. Kaggle source 공급 방식

### 3.1 Internet과 public Git 접근이 가능한 경우

Kaggle environment variable에 다음 값을 설정한다.

```text
CELL_MSCA_SOURCE_MODE=git
CELL_MSCA_GIT_SHA=<실행할 정확한 commit SHA>
```

notebook은 public repository를 `/kaggle/working/cell-msca-source`에 clone한 뒤
`git checkout --detach <SHA>`를 실행한다. branch tip이 이동해도 지정 SHA가
바뀌지 않는다.

### 3.2 Git 접근이 불가능한 경우

repository snapshot을 private Kaggle dataset으로 첨부한다. dataset slug를 코드에
고정하지 않는다. 첨부 root에는 최소한 다음 경로가 있어야 한다.

```text
src/cell_msca/
configs/kaggle_synthetic_smoke.json
tests/
GIT_COMMIT_SHA.txt
SOURCE_TREE_MANIFEST.json
```

환경 변수는 다음과 같다.

```text
CELL_MSCA_SOURCE_MODE=attached
CELL_MSCA_GIT_SHA=<snapshot을 만든 정확한 commit SHA>
CELL_MSCA_SOURCE_MANIFEST_SHA256=<trusted SOURCE_TREE_MANIFEST.json SHA-256>
```

SHA 또는 manifest 파일명이 다르면 `CELL_MSCA_SOURCE_SHA_FILE`과
`CELL_MSCA_SOURCE_MANIFEST`에 절대 경로를 지정한다. notebook은
`/kaggle/input`의 바로 아래에서 `src/cell_msca`를 포함한 유일한 dataset을 찾은 뒤
그 code snapshot을 `/kaggle/working/cell-msca-source`로 복사한다. test, import,
`compileall`은 working copy에서만 실행하므로 read-only input에는 쓰지 않는다.

신뢰할 수 있는 clean checkout에서 packaging manifest를 생성하는 예시는 다음과 같다.
출력 파일은 repository 밖의 packaging directory에 둔다.

```powershell
$commit = git rev-parse HEAD
$env:PYTHONPATH = "src"
python -c "from cell_msca.kaggle_runner import write_source_tree_manifest; print(write_source_tree_manifest('.', 'C:/package/SOURCE_TREE_MANIFEST.json', git_commit_sha='$commit'))"
```

출력된 SHA-256을 `CELL_MSCA_SOURCE_MANIFEST_SHA256`으로 별도 전달한다. 파일 변경,
누락, 추가 또는 manifest 자체 변경이 있으면 runner는 실행 전에 중단한다.

## 4. 정확한 로컬 synthetic 명령

PowerShell에서 repository root를 현재 directory로 둔다. output은 repository 밖의
새 directory를 사용한다.

```powershell
$commit = git rev-parse HEAD
$output = "C:\cell-msca-validation-smoke"
$variants = @("token_no_attention", "forward", "reverse", "bidirectional")

foreach ($variant in $variants) {
  python -m cell_msca.kaggle_runner `
    --variant $variant `
    --train-seed 3407 `
    --config configs/kaggle_synthetic_smoke.json `
    --data-path generated `
    --output-path $output `
    --device cpu `
    --expected-git-sha $commit `
    --repository-root .
}
```

`src` layout package가 editable install 상태가 아니라면 먼저 현재 shell에만 다음을
설정한다.

```powershell
$env:PYTHONPATH = "src"
```

이미 동일 experiment ID directory가 있으면 runner가 즉시 중단한다. 삭제 또는
덮어쓰기 대신 새 frozen configuration이나 별도 output root를 사용한다.

## 5. 정확한 Kaggle CLI 계약

notebook이 실행하는 CPU 명령은 다음과 동일하다.

```bash
python -m cell_msca.kaggle_runner \
  --variant forward \
  --train-seed 3407 \
  --config /kaggle/working/cell-msca-source/configs/kaggle_synthetic_smoke.json \
  --data-path generated \
  --output-path /kaggle/working/cell-msca-synthetic-validation \
  --device cpu \
  --expected-git-sha <EXACT_GIT_SHA> \
  --repository-root /kaggle/working/cell-msca-source \
  --kaggle
```

attached code dataset에서는 다음 인자를 추가한다.

```text
--source-git-sha-file /kaggle/input/<attached-code>/GIT_COMMIT_SHA.txt
--source-tree-manifest /kaggle/input/<attached-code>/SOURCE_TREE_MANIFEST.json
--expected-source-manifest-sha256 <TRUSTED_MANIFEST_SHA256>
```

notebook은 CPU에서 네 variant를 모두 실행한다. CUDA가 실제로 사용 가능한 경우에만
동일한 네 frozen CUDA configuration을 추가로 실행한다. CUDA가 없으면 skip 사실을
출력하며 오류로 취급하지 않는다.

## 6. 향후 project-data validation 경로

project-data config는 `data.mode=npz`, persistent split CSV/metadata 경로, 정확한
data/split/split-config/preprocessing/configuration hash를 포함해야 한다. `--data-path`는
첨부된 read-only NPZ dataset root를 가리킨다. runner는 다음 기존 경로만 호출한다.

1. metadata-bearing NPZ를 `CellDataset`으로 검증;
2. `load_persistent_split`을 통한 split metadata/data hash 검증;
3. `BaselineDataProtocol.from_dataset`을 통한 train-cell-only preprocessing;
4. `TuningData`의 train 및 validation materialization;
5. `fit_cell_msca`의 median-checkpoint 및 post-checkpoint inverse 선택;
6. 공통 evaluator와 prediction CSV metric round-trip;
7. 안전한 selected checkpoint 저장 및 `weights_only=True` reload.

runner에는 test split을 선택하는 CLI 인자가 없다. config의 stage 또는 allowed split에
`test`를 추가하면 loading 단계에서 `TestEvaluationBlockedError`가 발생한다.

## 7. output artifact schema

각 실행 directory 이름은 owner, variant, train seed, split seed, device,
data/split/split-config/preprocessing/configuration/Git hash로 계산한 deterministic
experiment ID다. 다음 일곱 파일만 생성한다.

| 파일 | 분류 | 내용 |
| --- | --- | --- |
| `resolved_config.json` | engineering-only | CLI override가 적용된 frozen training/config/hash 계약 |
| `run_manifest.json` | engineering-only | owner, ID, model, seed, hash, runtime, 시간, 상태, test gate |
| `environment.json` | engineering-only | Python, PyTorch, NumPy, LightGBM, CUDA, GPU 정보 |
| `validation_metrics.json` | validation-only | original-unit primary와 log/Spearman secondary metric |
| `validation_predictions.csv` | validation-only | cell ID와 원 단위/log 단위 true/prediction |
| `selected_checkpoint.pt` | validation-only | 기존 validation-selected checkpoint contract |
| `execution.log` | engineering-only | credential을 포함하지 않는 단계·상태 기록 |

`run_manifest.json`의 `artifacts` mapping이 모든 파일의 분류를 명시한다. 정상 종료는
`status=completed`, 예외 종료는 `status=failed`로 원자적으로 갱신된다. 두 경우 모두
`test_subset_materialized=false`를 유지한다.

## 8. 협업 workflow

1. coordinator가 data, split, preprocessing, configuration, Git hash를 동결한다.
2. coordinator가 `configs/experiment_assignment.example.json` 형식으로 owner별 고유
   experiment ID와 output location을 배정한다.
3. 중복 experiment ID는 assignment validator가 거부한다.
4. owner는 배정된 frozen config와 seed를 그대로 실행한다. 개인 판단으로 learning
   rate, loss, architecture width, inverse rule 또는 seed를 바꾸지 않는다.
5. owner는 `run_manifest.json`의 모든 hash와 `status=completed`를 확인한다.
6. 공유 시에는 일곱 artifact만 전달하고 NPZ, credential 또는 임시 synthetic input은
   포함하지 않는다.
7. 비교·선택은 validation artifact에서만 수행한다. test gate 개방은 별도 승인 및
   frozen-selection 절차가 필요하다.

## 9. 제출 전 점검

- notebook의 모든 `execution_count`가 `null`이고 `outputs`가 비어 있는가;
- repository test suite와 `compileall src tests`가 통과했는가;
- Git SHA와 config/data/split/preprocessing hash가 manifest와 일치하는가;
- `/kaggle/input` 아래에 변경된 파일이 없는가;
- output에 NPZ, credential, token, notebook output 또는 임시 checkpoint가 없는가;
- `test_subset_materialized`가 모든 artifact에서 `false`인가.
