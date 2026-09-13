# Cell-MSCA Implementation Specification

- Status: Phases 1-3 complete; Phase 3.5 evidence audit integrated; Phase 4 architecture and synthetic validation authorized
- Initial audit date: 2026-08-15 (Asia/Seoul)
- Evidence-alignment date: 2026-08-23 (Asia/Seoul)
- Intended paper venue, authorship, and citation metadata: human confirmation required
- Scope: target and data contracts, leakage controls, model architecture, validation gates, and implementation order

## 1. Decision summary

The current 16x16 ViT MS-CA implementation is retained as a **legacy comparison only**. The main implementation will estimate an ODIAC-referenced monthly value for one 1 km grid cell using seven scalar features from that same cell. It will not use neighbouring cells, coordinates, hotspot classes, or policy-level hotspot claims.

The implementation order is fixed as follows:

1. common target transform, inverse transform, metrics, and uncertainty;
2. persistent split files and a true single-cell dataset;
3. simple baselines using the identical split and evaluator;
4. feature-token Cell-MSCA and architectural ablations;
5. experiment runner, three-seed repetition, and paper tables.

The current 36-month NPZ archive may be used as **`v1_legacy ODIAC-derived zonal mean`** for code development and historical reproduction. It must not be presented as a confirmed `tC/cell/month` or `tCO2/cell/month` total until the ODIAC raster-to-grid operation, source release, and target unit are verified. A future `v2_corrected` dataset must use a documented mass-preserving or density-preserving raster operation chosen according to verified source metadata.

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

Phase 3.5 independently checked the important target, prior-work, feature-token, cross-attention, spatial-validation, transformation, loss, and metric claims against primary papers and official documentation. `REFERENCE_AUDIT.md`, `CLAIM_EVIDENCE_MATRIX.md`, and `references.bib` are the governing evidence record. Dataset-release claims still require the exact project source files and cannot be inferred from the current ODIAC website alone.

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

The repository preserves a flat 11-file 16x16 training/evaluation implementation as legacy code:

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

The active single-cell implementation is isolated under `src/cell_msca/`, with synthetic unit and integration tests under `tests/`. The flat files above must remain byte-identical to the Phase 3 tag unless a separately approved legacy-maintenance task is opened.

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
- Pollution/environment order: `no2_mean`, `so2_mean`, `co_mean`;
- Socio-infrastructure order: `nightlight_mean`, `urban_fraction`, `power_plant_count`, `fossil_capacity_mw`;
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

### 5.1 `v1_legacy ODIAC-derived zonal mean` target

Direct code evidence defines the current target as the arithmetic mean of ODIAC raster values selected by `all_touched=True`:

\[
y^{legacy}_{g,t}=\frac{1}{|P(g)|}\sum_{p\in P(g)}E_{p,t}.
\]

This value must be named **`v1_legacy ODIAC-derived zonal mean`**. It must not be interpreted as a confirmed target-cell monthly total. The exact ODIAC release is unresolved until the 36 source filenames, byte checksums, and source raster headers are recovered. The `v1_legacy` metadata must therefore use the conservative unit string `ODIAC native value; exact target-cell interpretation unverified` until resolved.

### 5.2 `v2_corrected` target candidate

If the source pixel stores a monthly total per source cell, a mass-preserving target-cell total is:

\[
y_{g,t}=\sum_p E_{p,t}\frac{A(p\cap g)}{A(p)}.
\]

The implementation must validate an appropriate conservation identity over the covered region and handle clipped boundary cells explicitly. If the source raster instead stores a density, the correct aggregation is different; the exact ODIAC metadata decides which branch applies. No `v2_corrected` data may be generated by guessing this branch.

Every source and derived target manifest must record: the exact release identifier; all 36 filenames and SHA-256 checksums; CRS; affine transform; raster shape and resolution; spatial extent; nodata value and handling; temporal coverage; source variable name and unit; carbon-versus-CO2 convention; reprojection and resampling operation; target-grid geometry; boundary-cell policy; aggregation equation; and derived artifact checksum. Any tC-to-tCO2 conversion requires an explicit documented factor and may not be inferred from value magnitude.

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

For cell `g` and month `t`, the stream names and feature order are fixed:

\[
x^P_{g,t}=[NO_2,SO_2,CO]\in\mathbb{R}^{3}
\quad\text{(pollution/environment)},
\]

\[
x^I_{g,t}=[Nightlight,UrbanFraction,PowerPlantCount,FossilCapacity]\in\mathbb{R}^{4}
\quad\text{(socio-infrastructure)}.
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
z_f^{(0)}=\tilde x_f w_f+b_f.
\]

Thus

\[
Z_P\in\mathbb{R}^{3\times d},\qquad Z_I\in\mathbb{R}^{4\times d}.
\]

Compressing each stream into one token before attention is forbidden because attention over a single key has a constant softmax weight of one.

### 6.4 Q, K, and V

The direction labels are experimental hypotheses, not domain truths. For the forward `I <- P` direction, Q is socio-infrastructure and K/V are pollution/environment:

\[
Q_I^h=LN(Z_I)W_Q^h,\quad K_P^h=LN(Z_P)W_K^h,\quad V_P^h=LN(Z_P)W_V^h,
\]

\[
H_{I\leftarrow P}^h=softmax\left(\frac{Q_I^h(K_P^h)^T}{\sqrt{d_h}}\right)V_P^h.
\]

The attention matrix has shape `[batch, heads, 4, 3]`.

For the reverse `P <- I` direction, `Q=Z_P` and `K=V=Z_I`; its attention matrix has shape `[batch, heads, 3, 4]`.

The current legacy 16x16 model uses only the analogous `I <- P` direction over spatial tokens. Neither project files nor literature establish that this direction is physically privileged. Forward, reverse, bidirectional, token/no-attention, and concat-MLP comparisons are mandatory, and attention weights must not be interpreted causally or as source attribution.

### 6.5 Fusion and head

The frozen Phase 4 implementation contract is pre-normalized residual processing. Each stream first applies feature-specific numerical tokenization, followed by the same number of independent pre-LayerNorm self-attention and feed-forward residual blocks and a final LayerNorm. A directional cross-attention block applies LayerNorm separately to query and context tokens, multi-head attention, a residual addition to the query stream, a pre-LayerNorm feed-forward residual, and final LayerNorm. All attention modules use batch-first tensors and operate independently within each sample; no operation may attend across the batch dimension.

For each direction, the cross-attention residual is:

\[
U_I=Z_I+Dropout\left(Concat_h(H^h_{I\leftarrow P})W_O\right),
\qquad
\tilde Z_{I\leftarrow P}=LN\left(U_I+FFN(LN(U_I))\right),
\]

with the symmetric expression for `P <- I`.

Every token variant uses the same two-stream pooled width and the same regression
head. Let `Pool` denote mean pooling over the feature-token dimension. The four
representations are:

\[
h_{no-cross}=Concat(Pool(Z_P),Pool(Z_I)),
\]

\[
h_{forward}=Concat(Pool(Z_P),Pool(\tilde Z_{I\leftarrow P})),
\]

\[
h_{reverse}=Concat(Pool(\tilde Z_{P\leftarrow I}),Pool(Z_I)),
\]

and

\[
h_{bidirectional}=Concat(Pool(\tilde Z_{P\leftarrow I}),Pool(\tilde Z_{I\leftarrow P})).
\]

Each representation is in `R^(2*d_model)` and passes through an otherwise
identical scalar regression head. The tokenizer, independent stream encoders,
pooling contract, and regression head are matched. The variants are not fully
parameter matched: forward and reverse add one cross-attention block, while the
bidirectional variant adds two. These capacity differences must be reported and
treated as a limitation when interpreting ablations.

For the frozen default architecture (`d_model=32`, `num_heads=4`, one stream
encoder layer, `ffn_multiplier=2`, and `head_hidden=32`), the trainable parameter
counts are:

| Variant | Total parameters | Added versus token/no-cross |
| --- | ---: | ---: |
| token/no-cross | 19,905 | 0 |
| forward `I <- P` | 28,577 | 8,672 (one cross-attention block) |
| reverse `P <- I` | 28,577 | 8,672 (one cross-attention block) |
| bidirectional | 37,249 | 17,344 (two cross-attention blocks) |

These totals must be recalculated if any architecture dimension changes. They do
not establish parameter-matched ablations; they quantify the remaining capacity
confound after matching the tokenizer, stream encoders, pooling, and head.

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

The paper's primary estimand remains unresolved. Median inverse naturally targets a conditional median on the transformed model scale, whereas Duan smearing is a mean-retransformation correction under its residual assumptions. Checkpoint selection for every log neural model uses median-inverse validation original-unit MAE; only after that checkpoint is fixed may median and train-residual-only Duan inverses be compared on validation data. The selected inverse is frozen before test access. This engineering rule does not itself resolve whether the paper's scientific estimand should be a conditional median, conditional mean, or regional total; that decision requires human confirmation and target-unit evidence.

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

The paper-facing target support is nonnegative. Every model therefore applies the
explicit, versioned policy `nonnegative_max_zero_v1` to its final original-unit
prediction: `final_prediction = max(0, unprojected_prediction)`. Finite-value
validation occurs before projection, the true target is never altered, and every
result records the number, fraction, and minimum of pre-projection negative values.
For original-target models, secondary log predictions are computed from the projected
original prediction. For log-target models, the native log prediction is retained for
log-space evaluation and only its original-unit inverse is projected. All
original-unit validation decisions, including early stopping, checkpoint selection,
median/Duan inverse selection, and final model selection, use this same policy.

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

The block and buffer distances must be selected from empirical spatial autocorrelation diagnostics rather than the legacy 16x16 patch size. The protocol must estimate target and predictor spatial dependence using training/development cells only, inspect variograms or correlograms and anisotropy, predefine candidate distances, test multiple block-grid origins, and report retained/dropped spatial coverage. Validation or test performance must not be used to choose the distance.

The chosen block split must be persisted, its actual train/validation/test/dropped cell counts recorded, and all pairwise minimum Chebyshev distances verified. The existing `block_size=24`, `buffer_cells=16` split is a development feasibility artifact only; it is not a research-justified final robustness split.

### 8.3 Optional temporal holdout

2022-2023 training/development with 2024 holdout is optional. If used, it evaluates a different generalization target and must be reported separately. The paper must not call the cell-fixed task forecasting.

## 9. Required model set

### 9.1 Core ACK set

1. train-mean predictor;
2. raw LightGBM;
3. log1p LightGBM;
4. Tweedie LightGBM;
5. Concat-MLP;
6. two-stream token encoder without cross-attention;
7. forward `I <- P` Cell-MSCA;
8. reverse `P <- I` Cell-MSCA;
9. bidirectional Cell-MSCA.

TabM or FT-Transformer is optional after the above set is complete. The architecture is justified only if its gain over LightGBM, Concat-MLP, and the no-attention ablation is larger than seed variation.

### 9.2 Mandatory construction-overlap ablation

- O0: all seven features;
- O1: remove nightlight only;
- O2: remove power-plant count and fossil capacity only;
- O3: remove nightlight, power-plant count, and fossil capacity together.

Run these on the selected model using the same split and evaluator. Nightlight directly overlaps ODIAC's documented non-point spatial disaggregation proxy; power-plant variables overlap conceptually with ODIAC point-source allocation, while exact source-dataset identity remains unresolved. These ablations are mandatory but cannot make ODIAC an independent field measurement.

## 10. Current code versus normative specification

| Status | Verified repository state | Remaining requirement |
|---|---|---|
| Complete | shared target transform, median and Duan inverse, original/log metrics, and separate cell-cluster bootstrap under `src/cell_msca/` | retain prediction-file metric reproduction tests |
| Complete | true one-cell `CellDataset`, fixed 3/4 feature order, persistent cell-fixed split, separated seeds, and train-only runtime preprocessing | rebuild raw pre-imputation features for a corrected data version |
| Complete | train mean, raw/log1p/Tweedie LightGBM, Concat-MLP, shared evaluator, validation-only tuning, and frozen test gate | run real baselines only after data-provenance blockers are resolved |
| Development only | persistent buffered-block feasibility path records ratios and minimum distances | replace development distance choices with the empirical autocorrelation protocol in section 8.2 |
| Phase 4 | feature tokenizer, token/no-attention, forward, reverse, and bidirectional architectures | synthetic validation first; no project-data training in Phase 4.1 |
| Blocked | legacy source tensors contain month-wide pre-split imputation | recover raw feature extracts and create a new immutable data version |
| Blocked | exact source ODIAC files and headers are absent | recover the 36-file manifest and determine the correct v2 aggregation branch |
| Human decision | public authorship, venue, citation metadata, target unit, and mean-versus-median estimand | record explicit approval; do not infer from legacy material |

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
- gradients reach both feature streams, Q/K/V projections, and the regression head;
- batch permutation and per-sample isolation tests demonstrate that attention never mixes samples;
- deterministic CPU initialization and prediction are verified;
- selected-model checkpoint save/reload produces identical predictions;
- AdamW excludes bias and normalization parameters from weight decay;
- every variant passes a `2*d_model` pooled representation to the same regression head;
- tokenizer, stream encoders, pooling contract, and regression head are matched, while the parameters added by one or two cross-attention blocks are reported as an ablation limitation;
- parameter counts, unavoidable capacity differences, forward/backward shapes, device, runtime versions, configuration hash, split hash, data hash, preprocessing hash, Git commit SHA, and the tracked-plus-untracked dirty-state policy are logged for every variant;
- actual PyTorch forward, backward, and checkpoint smoke tests pass before Phase 4.1 is declared complete.

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
5. Is the paper's primary estimand a conditional median, conditional mean, or a regional total? Validation may select an inverse mode for a fixed checkpoint but cannot answer the scientific estimand question by itself.
6. What empirical spatial autocorrelation range and anisotropy justify the final block and buffer candidates?
7. Is buffered block robustness mandatory for the intended submission, or a documented later extension if infeasible?
8. Should the legacy 16x16 result appear in a clearly separated historical appendix or be omitted entirely?
9. What authorship order, affiliations, paper title, venue, and citation metadata have all contributors approved?

## 15. Current authorized code-change stage

Phases 1-3 are complete and protected by the Phase 3 baseline tag. Phase 4.0 aligns repository claims with the Phase 3.5 audit. Phase 4.1 may add only the single-cell feature tokenizer, token/no-attention and directional architecture variants, reusable neural runtime/checkpoint utilities, and synthetic tests.

Phase 4.1 must not run full `v1_legacy` training, open the test gate, create paper-facing performance results, reconstruct source rasters by assumption, generate `v2_corrected` data, alter persistent split artifacts, or modify the flat legacy `src/*.py` files.

## 16. Verified implementation and research boundary

### Confirmed implementation facts

- Phase 1 shared inverse, metrics, bootstrap, and prediction round-trip contracts are tested;
- Phase 2 single-cell data, feature metadata, persistent splits, separated seeds, and preprocessing provenance contracts are tested;
- Phase 3 baseline selection uses validation only, computes Duan from train residuals only, and keeps the test subset behind a frozen-selection gate;
- the current NPZ archive contains three pollution/environment and four socio-infrastructure channels, raw `label_reg`, and a stable mask according to the recovered project artifacts;
- the recovered preprocessing code computes the legacy target with zonal mean and `all_touched=True` and fills some missing features before splitting;
- flat legacy source and result-time source are not fully identical, so historical results cannot be assigned to the active package;
- Phase 3.5 documents the literature evidence, project evidence, hypotheses, and unresolved claims.

### Research boundary before project-data training

Only unit tests, synthetic integration tests, architecture parameter accounting, and isolated runtime smoke tests are authorized. Project-data training remains blocked by target provenance, physical aggregation, upstream imputation, estimand, and spatial-protocol decisions. The test split remains closed until the data version, split, preprocessing, configuration, selected checkpoint, inverse mode, and seeds are frozen under a reviewed protocol.
