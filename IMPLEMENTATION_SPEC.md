# Cell-MSCA Implementation Specification (Stage 0)

- Status: implementation baseline; no training authorized in this stage
- Audit date: 2026-08-15 (Asia/Seoul)
- Primary target: ACK 2026 fall submission
- Scope: code and artifact audit, mathematical contract, implementation order
- Existing source code was not modified. This document is the only new project file.

## 1. Decision summary

The current 16x16 ViT MS-CA implementation is retained as a **legacy comparison only**. The main implementation will estimate an ODIAC-referenced monthly value for one 1 km grid cell using seven scalar features from that same cell. It will not use neighbouring cells, coordinates, hotspot classes, or policy-level hotspot claims.

The implementation order is fixed as follows:

1. common target transform, inverse transform, metrics, and uncertainty;
2. persistent split files and a true single-cell dataset;
3. simple baselines using the identical split and evaluator;
4. feature-token Cell-MSCA and architectural ablations;
5. experiment runner, three-seed repetition, and paper tables.

The current 36-month NPZ archive may be used as `v1_legacy` for code development and historical reproduction. It must not be presented as a corrected physical target until the ODIAC raster-to-grid operation and target unit are verified. A future `v2_corrected` dataset must use a documented mass-preserving or density-preserving raster operation chosen according to the verified source unit.

## 2. Evidence classes used in this specification

### 2.1 Direct file evidence

The following claims were checked directly in local files:

- current flat source tree under `src/`;
- `outputs/MSCA_FINAL_RESEARCH_PLAN_KO.md`;
- `README_preprocessing.md`;
- `carbon-hotspot-detection-main.zip`, extracted only into an audit directory;
- 36 existing NPZ files in `2022_2024_npz.zip`;
- result JSON/CSV/NPZ files and result-time source snapshot;
- checkpoint archive structure and parameter names.

### 2.2 External or literature-based claims

ODIAC's official unit, the conceptual interpretation of ODIAC, and comparisons with OpenCarbon and other studies come from the cited material in `MSCA_FINAL_RESEARCH_PLAN_KO.md`. They were not independently re-verified during this local Stage 0 audit. Before paper submission, each such claim must be linked to the exact ODIAC version README or primary paper.

### 2.3 Unverified items

The following remain unverified even though reconstructed code now exists:

- whether the uploaded preprocessing code is the exact code that generated the 36 NPZ files;
- the exact ODIAC source GeoTIFF checksums and metadata used for those files;
- Google Earth Engine image counts, QA/cloud filters, and execution logs from the original run;
- whether every monthly intermediate CSV can be reproduced byte-for-byte;
- whether `label_reg` should be interpreted as a target-cell total, a source-pixel mean, or a density after reprojection;
- licenses and redistribution conditions for every derived artifact.

## 3. Artifact inventory and lineage

### 3.1 Current project

The current project is a flat 11-file training/evaluation implementation:

- `src/model.py`
- `src/spatial_grid_dataset.py`
- `src/splits_v2.py`
- `src/train.py`
- `src/metrics.py`
- `src/evaluation.py`
- `src/inference.py`
- `src/baselines.py`
- `src/diagnostics.py`
- `src/fast_data.py`
- `src/make_figures.py`

It has no `tests/` directory and is not currently a Git repository.

### 3.2 Newly supplied project ZIP

`carbon-hotspot-detection-main.zip` is not a duplicate of the current project. It contains:

- configuration-driven preprocessing under `src/preprocessing/`;
- the entry point `scripts/run_preprocessing.py`;
- a Delhi 2022-2024 YAML configuration;
- a top-level older training script;
- a separate package-form MS-CA implementation under `src/ms-ca/`;
- tests for that separate package.

The ZIP therefore contains at least three overlapping code paths. It must be treated as a **preprocessing and legacy-code donor**, not copied wholesale into the new repository.

### 3.3 Existing data archive

Direct inspection of `2022_2024_npz.zip` confirmed:

- 36 files, from 2022-01 through 2024-12;
- `stream_a`: `[3, 136, 130]`, `float32`;
- `stream_b`: `[4, 136, 130]`, `float32`;
- `label_reg`: `[136, 130]`, `float32`;
- `label_cls`: `[136, 130]`, `int64`, legacy metadata;
- `mask`: `[136, 130]`, `uint8`;
- Stream A order: `no2_mean`, `so2_mean`, `co_mean`;
- Stream B order: `nightlight_mean`, `urban_fraction`, `power_plant_count`, `fossil_capacity_mw`;
- the same mask in every month;
- 17,157 valid cells per month and 617,652 cell-month samples;
- no negative `label_reg` values;
- 4,644 zero targets;
- `label_reg` minimum 0, maximum 94,549.625, mean about 152.3334, median about 26.7700.

Archive SHA-256 values recorded during the audit:

| Artifact | SHA-256 |
|---|---|
| `carbon-hotspot-detection-main.zip` | `50CBFA18792D7813182A6F1B5E57B8D5B17BE4F736472DCED989BED13C9F7E42` |
| `2022_2024_npz.zip` | `5E01B3BC58FA2ADEBBE24AA2B6C3D2D2B93D274ADC1C2686A37C62F57A5D93ED` |
| `results.zip` | `F1E2677DF4850ECE052F9265447C426B333166DA0A40554977E44EB0CF65F8FF` |
| `best_model.zip` | `5126F1B5D908CDFEF1CF56A9DAA62D9BBF3027292370CF5D38960D95BC29BDCE` |

### 3.4 Existing result lineage

The result archive contains a source snapshot, but only `spatial_grid_dataset.py`, `baselines.py`, and `fast_data.py` are byte-identical to the current copies. The other result-time files differ from the current project. Existing numerical results must therefore be tied to the source snapshot inside the result archive, not to the current source tree.

The directly recorded cell-fixed test result in `eval_cell.json` is:

| Space | MAE | RMSE | R2 | Spearman |
|---|---:|---:|---:|---:|
| log1p | 0.158924 | 0.264049 | 0.959521 | 0.977728 |
| original | 45.059739 | 1149.418749 | 0.066603 | 0.977728 |

These are legacy results, not final Cell-MSCA results. The confidence intervals stored in that file were calculated from log-space arrays even though the new protocol requires original-space intervals for headline metrics.

## 4. Current data flow

The recovered preprocessing path is:

1. `grid.py` creates 1,000 m cells in EPSG:32643, clips them to the configured rectangular boundary, assigns `row`, `col`, and `grid_id`, and exports EPSG:4326 geometry.
2. `gee_extract.py` filters monthly GEE collections by date, selects one band, computes an image-collection mean, and applies `reduceRegions` using the configured mean reducer.
3. `vector_features.py` assigns WRI power-plant points to grid cells and produces plant count and fossil capacity.
4. `cleaning.py` fills missing raster feature values with a month-wide mean, median, or zero before the model split exists.
5. `raster_label.py` reprojects grid polygons to the ODIAC raster CRS and runs `rasterstats.zonal_stats(..., stats=["mean", "max", "min"], all_touched=True)`.
6. `merge.py` joins features and labels on `(grid_id, date)`.
7. `tensor_builder.py` stores raw `label_reg` in each NPZ. It does not store `label_reg_log` in the NPZ.
8. the current dataset optionally applies `log1p` in memory, builds one sample for every valid `(month,row,col)`, and extracts a 16x16 crop around that cell.
9. `train.py` builds the split, applies train-cell feature normalization, trains the 16x16 model, and selects a checkpoint using a training-space metric.
10. `evaluation.py` and `inference.py` rebuild the split and invert predictions.

## 5. Normative target specification

### 5.1 `v1_legacy` target

Direct code evidence defines the current target as the arithmetic mean of ODIAC raster values selected by `all_touched=True`:

\[
y^{legacy}_{g,t}=\frac{1}{|P(g)|}\sum_{p\in P(g)}E_{p,t}.
\]

This value must be described as an **ODIAC-derived zonal mean** unless the exact raster alignment and source unit demonstrate that it is also a valid target-cell monthly total. The `v1_legacy` metadata must therefore use a conservative unit string such as `ODIAC native value; exact target-cell interpretation unverified` until resolved.

### 5.2 `v2_corrected` target candidate

If the source pixel stores a monthly total per source cell, a mass-preserving target-cell total is:

\[
y_{g,t}=\sum_p E_{p,t}\frac{A(p\cap g)}{A(p)}.
\]

The implementation must validate an appropriate conservation identity over the covered region and handle clipped boundary cells explicitly. If the source raster instead stores a density, the correct aggregation is different; the exact ODIAC metadata decides which branch applies.

### 5.3 Target transform

After confirming the physical unit, define a reference scale

\[
c=1\;\text{target unit}
\]

and

\[
z_{g,t}=\ln\left(1+\frac{y_{g,t}}{c}\right).
\]

The uncorrected inverse is

\[
\hat y^{median}_{g,t}=c\left(e^{\hat z_{g,t}}-1\right).
\]

Negative raw targets must raise a validation error. The current behavior that silently clips them to zero before `log1p` must not be used in the final pipeline.

### 5.4 Duan smearing

Using train residuals only,

\[
r_i=z_i-\hat z_i,\qquad S=\frac{1}{n_{train}}\sum_i e^{r_i},
\]

and

\[
\hat y^{mean}_{g,t}=c\left(e^{\hat z_{g,t}}S-1\right).
\]

All evaluation and inference code must call one shared inverse-transform function. The current direct implementation `(expm1(pred) * S)` is not equivalent. The inverse mode must be chosen using validation data and recorded before the test split is opened.

## 6. Normative input and Cell-MSCA specification

### 6.1 Inputs

For cell `g` and month `t`:

\[
x^A_{g,t}=[NO_2,SO_2,CO]\in\mathbb{R}^{3},
\]

\[
x^B_{g,t}=[NTL,Urban,PlantCount,FossilCapacity]\in\mathbb{R}^{4}.
\]

Coordinates, grid IDs, neighbouring-cell values, and `label_cls` are not model inputs.

Every sample must also carry metadata that is not fed into the model: `date`, `row`, `col`, stable `cell_id`, `data_version`, feature order, and target unit.

### 6.2 Train-only preprocessing

For each feature `f`, fit preprocessing parameters on train cells only and apply them unchanged to validation and test:

\[
\tilde x_f=\frac{x_f-\mu_f^{train}}{\sigma_f^{train}+\epsilon}.
\]

For `v2_corrected`, missing-value statistics must also be derived from train cells only. The existing month-wide feature mean is already baked into `v1_legacy`; this limitation must be recorded rather than hidden.

### 6.3 Feature tokens

Each scalar remains a separate token:

\[
z_f^{(0)}=w_f\tilde x_f+b_f+e_f+e_{group(f)}.
\]

Thus

\[
Z_A\in\mathbb{R}^{3\times d},\qquad Z_B\in\mathbb{R}^{4\times d}.
\]

Compressing each stream into one token before attention is forbidden because attention over a single key has a constant softmax weight of one.

### 6.4 Q, K, and V

For the `B <- A` direction:

\[
Q_B^h=LN(Z_B)W_Q^h,\quad K_A^h=LN(Z_A)W_K^h,\quad V_A^h=LN(Z_A)W_V^h,
\]

\[
H_{B\leftarrow A}^h=softmax\left(\frac{Q_B^h(K_A^h)^T}{\sqrt{d_h}}\right)V_A^h.
\]

The attention matrix has shape `[batch, heads, 4, 3]`.

For the `A <- B` direction, `Q=Z_A` and `K=V=Z_B`; its attention matrix has shape `[batch, heads, 3, 4]`.

The current 16x16 model uses only `B <- A`, where Q is the 16 infrastructure spatial tokens and K/V are the 16 environmental spatial tokens. The source code justifies this as an attribution interpretation. It contains no evidence that Q was selected because coordinates are fixed. Coordinates are not passed to the current model. Attention direction is therefore an experimental design choice and must be tested through `B <- A`, `A <- B`, and bidirectional ablations.

### 6.5 Fusion and head

For each direction:

\[
\tilde Z_{B\leftarrow A}=LN\left(Z_B+Concat_h(H^h_{B\leftarrow A})W_O\right),
\]

with the symmetric expression for `A <- B`.

The bidirectional representation is

\[
h=Concat(MeanPool(\tilde Z_{B\leftarrow A}),MeanPool(\tilde Z_{A\leftarrow B})),
\]

followed by a scalar regression head. The one-way models pool only their updated query stream.

## 7. Loss, selection, and evaluation contract

### 7.1 Loss candidates

The required neural-model candidates are log-L1 and log-Huber:

\[
\mathcal L_{log-L1}=\frac{1}{N}\sum_i|\hat z_i-z_i|.
\]

For `e_i = hat(z)_i - z_i`, Huber loss is

\[
\mathcal L_{Huber}=\frac1N\sum_i
\begin{cases}
\frac12 e_i^2,& |e_i|\le\delta,\\
\delta(|e_i|-\frac12\delta),& |e_i|>\delta.
\end{cases}
\]

Optimizer, dropout, and weight decay remain tuned configuration values rather than scientific contributions. AdamW may be retained, but weight decay must be searched in the planned small range and excluded from bias and normalization parameters.

### 7.2 Model selection

- use validation original-unit MAE as the primary checkpoint-selection metric;
- use original-unit RMSE and bias as secondary diagnostics;
- freeze model, loss, inverse mode, and hyperparameters before final test evaluation;
- evaluate test once per pre-registered final seed/configuration;
- do not select any option using test results.

### 7.3 Headline metrics

\[
MAE=\frac1N\sum_i|\hat y_i-y_i|,
\]

\[
RMSE=\sqrt{\frac1N\sum_i(\hat y_i-y_i)^2},
\]

\[
R^2=1-\frac{\sum_i(y_i-\hat y_i)^2}{\sum_i(y_i-\bar y)^2},
\]

\[
Bias=\frac1N\sum_i(\hat y_i-y_i).
\]

Original-unit MAE, RMSE, and R2 are primary. Log-space metrics and Spearman are secondary. Hotspot recall, top-10% classification, ROC-AUC, PR-AUC, and policy hotspot maps are excluded from the paper-facing evaluator.

### 7.4 Uncertainty

Bootstrap complete cells rather than individual cell-month samples. Original-unit predictions and targets must be passed to the original-unit bootstrap. Log-space intervals must be separately named and stored.

## 8. Split contract

### 8.1 Main split

- one stable `(row,col)` cell belongs to exactly one of train, validation, or test;
- all 36 months of that cell follow it;
- use a single persistent `split_seed`, initially 42;
- save `splits/cell_fixed_seed42.csv` with `row`, `col`, `cell_id`, and `split`;
- save the file SHA-256 and use the same file for every model and training seed.

The current `train.py` uses one `args.seed` for both split assignment and model initialization. That violates the planned three-seed protocol because seed 42, 2026, and 3407 would create different test sets. The new implementation must separate `split_seed` from `train_seed`.

### 8.2 Spatial robustness

A cell-fixed split prevents the same coordinate from crossing splits, but it does not remove correlation between neighbouring train and test cells. A buffered spatial block split is therefore a required robustness evaluation if feasible within the submission schedule. It is conceptually distinct from removing 16x16 input patches.

The block split must be persisted, its actual train/validation/test/dropped cell counts recorded, and the minimum train-to-held-out distance verified. The current randomized block code is a useful prototype but has not yet been validated on the final 1x1 experiment protocol.

### 8.3 Optional temporal holdout

2022-2023 training/development with 2024 holdout is optional. If used, it evaluates a different generalization target and must be reported separately. The paper must not call the cell-fixed task forecasting.

## 9. Required model set

### 9.1 Core ACK set

1. train-mean predictor;
2. raw LightGBM;
3. log1p LightGBM;
4. Tweedie LightGBM;
5. Concat-MLP;
6. two-stream no-cross-attention neural ablation;
7. `B <- A` Cell-MSCA;
8. `A <- B` Cell-MSCA;
9. bidirectional Cell-MSCA.

TabM or FT-Transformer is optional after the above set is complete. The architecture is justified only if its gain over LightGBM, Concat-MLP, and the no-attention ablation is larger than seed variation.

### 9.2 Feature-group ablation

- F1: NO2, SO2, CO;
- F2: NTL, Urban, PlantCount, Capacity;
- F3: all features except NTL, PlantCount, and Capacity;
- F4: all seven features.

Run these on the selected model using the same split and evaluator. Record that NTL and power-plant information may overlap with ODIAC's inventory construction; ablation reduces but does not eliminate the limitation that ODIAC is not an independent field measurement.

## 10. Current code versus normative specification

| Priority | Actual file evidence | Required state |
|---|---|---|
| P0 | `raster_label.py` stores `zonal_stats(..., mean, all_touched=True)` | verify source unit/alignment; implement a scientifically matched v2 aggregation |
| P0 | `evaluation.py` and `inference.py` multiply `expm1(pred)` by smearing | call the shared `exp(pred)*S-1` implementation |
| P0 | `evaluation.py` bootstraps training-space arrays | bootstrap original and log spaces separately |
| P0 | one seed controls split and model initialization | separate persistent `split_seed` and per-run `train_seed` |
| P0 | result-time source differs from current source | archive result-time source as legacy and rerun every final result |
| P0 | existing preprocessing code is not proven to be the actual NPZ generator | obtain execution provenance and source/intermediate checksums |
| P1 | current dataset always returns 16x16 crops | add a true scalar-feature `CellDataset` |
| P1 | current model creates 16 spatial tokens per stream | create 3 and 4 feature tokens without spatial patches |
| P1 | current model implements only Q=Stream B, K/V=Stream A | add reverse and bidirectional variants |
| P1 | D4 rotation/flip augmentation is enabled by default | remove it from the single-cell pipeline |
| P1 | checkpoint selection uses log-space R2 or MAE | select on validation original-unit MAE |
| P1 | scheduler contains three cosine cycles | replace with one documented cosine decay or a simpler fixed schedule |
| P1 | baseline file has only one LightGBM path plus simple linear models | add raw, log1p, Tweedie, Concat-MLP, and shared evaluation |
| P1 | preprocessing fills missing raster features using all cells in that month | fit statistical imputation on train cells for v2; document v1 leakage limitation |
| P1 | `tensor_builder.py` requires `label_cls` | make classification metadata optional and irrelevant to model contracts |
| P1 | current metrics and README emphasize top-10% hotspot recall | exclude hotspot outputs from paper-facing code and claims |
| P1 | current `README.md` calls log R2 the primary accuracy metric | make original-unit metrics primary and label existing numbers legacy |
| P1 | `README_preprocessing.md` says `label_reg_log` may be the tensor target | correct documentation: current NPZ contains raw `label_reg`; log1p occurs in the loader |
| P1 | current loader clips negative target values before log1p | fail validation on negative targets |
| P1 | current repository has no tests | add unit, integration, split, and smoke tests before full training |
| P2 | checkpoint lacks data archive hash, exact feature-order contract, target unit, and data version | store all provenance fields in run metadata and checkpoint |
| P2 | `pyproject.toml` lists one author while the paper is planned as co-first-author work | update project metadata only after both authors approve the public form |
| P2 | final plan section 8.2 omits buffered spatial robustness | add block robustness to the experiment protocol or document a schedule-based omission |

## 11. Reusable code and code to isolate

### 11.1 Reuse after tests

- core MAE/RMSE/R2/Bias calculations in `metrics.py`;
- the current corrected `duan_smearing()` and `apply_smearing()` formulas;
- cluster bootstrap concept in `bootstrap_ci_by_cell()`;
- cell-fixed assignment concept and overlap checks in `splits_v2.py`;
- train-only normalization concept and checkpoint metadata concept;
- center-cell feature extraction in `baselines.py`;
- preprocessing configuration, path helpers, grid construction, and pipeline entry point from the new ZIP;
- selected model/metric tests from the nested package after rewriting them for the new contracts.

### 11.2 Isolate as legacy

- 16x16 ViT encoders and spatial attention visualization;
- `FastPatchBatcher` and D4 augmentation;
- context `full/center_only/shuffle` experiments;
- hotspot figures and top-decile metrics;
- per-sample random split;
- duplicated top-level and nested training packages from the new ZIP.

## 12. Recommended file structure

Do not rewrite the flat legacy implementation in place. After creating the Git baseline, add a separate package:

```text
src/
  cell_msca/
    data.py                 # CellDataset and feature metadata
    splits.py               # persistent cell/block split files
    target.py               # transform, inverse, smearing
    metrics.py              # paper-facing metrics and bootstrap
    baselines.py            # LightGBM variants and train mean
    neural_baselines.py     # Concat-MLP and no-CA
    model.py                # feature tokenizer and Cell-MSCA variants
    train.py                # neural training only
    evaluate.py             # common evaluator
    experiment.py           # config/run orchestration
tests/
  test_target.py
  test_metrics.py
  test_splits.py
  test_cell_dataset.py
  test_models.py
  test_experiment_contract.py
configs/
  data_v1_legacy.yaml
  data_v2_corrected.yaml
  experiments/
splits/
docs/
  DATA_CARD.md
  EXPERIMENT_PROTOCOL.md
legacy/
  RESULT_CODE_MANIFEST.md
```

Existing source files remain untouched until the new package passes parity and contract tests.

## 13. Implementation phases and completion gates

### Phase 1: evaluation core

Implement target transforms, both inverse modes, raw/log metrics, and cell bootstrap.

Completion gates:

- perfect predictions give MAE=RMSE=0 and R2=1 in both spaces;
- for `pred_log=0` and `S=2`, smearing returns `1`, catching the old formula that returns `0`;
- original-space bootstrap receives original-space arrays;
- no hotspot field appears in paper-facing metric output;
- predictions saved to CSV reproduce the JSON metrics exactly.

### Phase 2: dataset and splits

Implement `CellDataset`, persistent split files, and train-only preprocessing.

Completion gates:

- sample feature shapes are `[3]` and `[4]`;
- no neighbour or coordinate tensor is returned to the model;
- feature names and order are validated against NPZ metadata;
- train/validation/test cell intersections are empty;
- changing `train_seed` does not change the split checksum;
- sentinel validation/test values cannot influence normalization or v2 imputation statistics;
- cell-fixed and block split cell counts and checksums are written to metadata.

### Phase 3: baselines

Implement the core baseline set using one evaluator.

Completion gates:

- all models consume the identical split file;
- all result rows contain data hash, split hash, config hash, train seed, target transform, inverse mode, and raw/log metrics;
- LightGBM early stopping uses validation only;
- no test metric is read during tuning.

### Phase 4: Cell-MSCA

Implement feature tokenizer, no-CA, both one-way directions, and bidirectional fusion.

Completion gates:

- token shapes are `[B,3,d]` and `[B,4,d]`;
- attention shapes are `[B,H,4,3]` and `[B,H,3,4]`;
- each attention row sums to one within tolerance;
- gradients reach feature projections, Q/K/V projections, and the regression head;
- a tiny subset can be overfit, demonstrating that the implementation can learn;
- parameter counts and forward shapes are logged for every variant.

### Phase 5: final experiment

- tune with split seed 42 and train seed 42;
- freeze configuration;
- repeat final candidates with train seeds 42, 2026, and 3407 on the identical split;
- run F1-F4 on the selected architecture;
- run buffered block robustness if feasible;
- open the test split only under the frozen protocol;
- report mean +/- standard deviation across seeds and cell-cluster confidence intervals.

## 14. Immediate decisions still required

1. Is the recovered preprocessing ZIP the exact code used for the July NPZ generation, or a later reconstruction?
2. Which ODIAC release and exact 36 GeoTIFF files were used?
3. Does the source raster value represent monthly total per source pixel or an areal density?
4. For clipped Delhi boundary cells, will v2 report target-cell total or area-normalized density?
5. Which inverse mode is primary for each loss? The choice must be frozen on validation.
6. Is buffered block robustness mandatory for ACK, or a documented post-submission extension if time is insufficient?
7. Should the legacy 16x16 result appear in an appendix or be omitted entirely?

## 15. First authorized code-change stage

The first code change must be **Phase 1 only**:

- create the new `cell_msca.target` and `cell_msca.metrics` modules;
- implement one shared inverse-transform path;
- implement separate original/log bootstrap outputs;
- add regression tests for the known smearing error;
- add prediction-to-metric round-trip tests;
- do not change the dataset, model, split, or training code in that commit.

This isolates evaluation correctness before any new model result is generated.

## 16. Validation report for Stage 0

### Confirmed facts

- the current NPZ archive has the expected 36 months, seven feature channels, raw `label_reg`, and stable mask;
- the recovered preprocessing code computes `label_reg` using zonal mean with `all_touched=True`;
- `label_reg_log` is created in CSV preprocessing but is not stored in the NPZ;
- the current model uses Q from Stream B and K/V from Stream A over 16 spatial tokens;
- the current split can keep a cell fixed across months and has a buffered-block prototype;
- train-only normalization is implemented in the current training path;
- missing-value filling in the recovered preprocessing occurs before splitting and can include held-out cells;
- existing evaluation/inference use an inconsistent smearing inverse;
- existing bootstrap confidence intervals are training-space intervals;
- current and result-time source code are not identical;
- the current project has no tests and no Git history.

### First verification after coding begins

Run only unit tests and small synthetic integration tests. Do not train on the 36-month data until Phase 1 and Phase 2 gates pass and a persistent split checksum exists.
