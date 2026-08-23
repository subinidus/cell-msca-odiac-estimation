# Cell-MSCA: single-cell ODIAC-referenced estimation

> [!IMPORTANT]
> This repository currently provides a leakage-controlled research framework and synthetic validation code. Phase 4 Cell-MSCA has not yet been evaluated on the project data, and no current Cell-MSCA performance result is available.

## Research task

The active task is **same-period, cell-level estimation** of an ODIAC-referenced inventory proxy for Delhi NCR. Each sample represents one grid cell in one month and uses seven scalar features from that same cell.

ODIAC is an inventory-based, spatially disaggregated emissions product. It is not a direct satellite CO2 measurement, physical ground truth, or independent field monitor. The current target must be named **`v1_legacy ODIAC-derived zonal mean`**. Its exact physical unit and the exact ODIAC release used to build the archived tensors remain unresolved.

This is not a forecasting task. Forecasting language requires a separately defined chronological future-month holdout.

## Current status

| Phase | Status | Scope |
|---|---|---|
| Phase 1 | Complete | Target transforms, median and Duan inverses, original/log metrics, cell-cluster bootstrap, prediction-file verification |
| Phase 2 | Complete | Single-cell dataset, feature-order checks, persistent cell-fixed split, train-only preprocessing contracts |
| Phase 3 | Complete | Train mean, LightGBM variants, Concat-MLP, shared evaluator, validation-only selection and frozen test gate |
| Phase 3.5 | Complete | Primary-source reference audit and claim-evidence matrix |
| Phase 4 | In progress | Feature-token Cell-MSCA architecture and synthetic-only validation; no project-data result |

See [RESEARCH_STATUS.md](RESEARCH_STATUS.md), [IMPLEMENTATION_SPEC.md](IMPLEMENTATION_SPEC.md), [REFERENCE_AUDIT.md](REFERENCE_AUDIT.md), and [CLAIM_EVIDENCE_MATRIX.md](CLAIM_EVIDENCE_MATRIX.md).

## Active model contract

The model input contains one cell only:

- pollution/environment stream: `NO2`, `SO2`, `CO` — shape `[batch, 3]`;
- socio-infrastructure stream: nightlight, urban fraction, power-plant count, fossil capacity — shape `[batch, 4]`.

Neighbouring cells, 16×16 patches, coordinates, row/column indices, hotspot labels, and `label_cls` are excluded from the active model input.

Directional cross-attention is an experimental hypothesis. Forward, reverse, bidirectional, token/no-attention, and concat-MLP variants must be compared before making an architectural claim. Attention weights are not causal or source-attribution evidence.

## Validation contract

The primary split is the persistent cell-fixed split in `splits/cell_fixed_seed42.csv`. It keeps all months of one cell in a single subset, but it does **not** establish spatial independence between neighbouring cells.

Buffered spatial evaluation is planned as a robustness analysis. Its block and buffer distances must be selected from empirical spatial autocorrelation diagnostics and sensitivity checks, not inherited from the legacy 16×16 patch size.

Headline metrics are original-unit MAE, RMSE, R², and Bias. Log-space metrics and Spearman correlation are secondary. The test subset remains closed until configuration, checkpoint, inverse mode, and validation selection are frozen.

## Reproducibility status

Data, checkpoints, generated predictions, and result figures are not distributed in Git. The repository includes code, tests, example configuration, immutable split artifacts, and their metadata.

Paper-facing reproduction remains blocked until all of the following are recovered or resolved:

1. the exact 36-file ODIAC source manifest, checksums, and raster headers;
2. source CRS, affine transform, resolution, nodata, time coverage, and unit metadata;
3. a physically valid boundary-cell and mass-preserving `v2_corrected` aggregation rule;
4. raw pre-imputation features so missing-value statistics can be fitted on train cells only;
5. a documented mean-versus-median estimand decision;
6. an empirical spatial-autocorrelation protocol.

The archived `v1_legacy` tensors may be used for engineering validation, but they must not be treated as a confirmed physical total or final paper dataset.

## Repository layout

```text
configs/                 portable validation-only baseline example
splits/                  persistent cell-fixed and development buffered splits
src/cell_msca/           active single-cell package
src/*.py                 immutable legacy 16×16 implementation
tests/                   synthetic unit and integration tests
docs/legacy/             historical poster-stage documentation
REFERENCE_AUDIT.md       literature and project-evidence audit
CLAIM_EVIDENCE_MATRIX.md claim-by-claim evidence classification
references.bib           verified bibliography
```

The earlier 16×16 README and its unreproduced numerical claims are preserved only in [docs/legacy/README_16x16_LEGACY.md](docs/legacy/README_16x16_LEGACY.md).

## Local validation

Python 3.10 or newer is declared by the package metadata. Install project dependencies in an isolated environment; do not place data or credentials in the repository.

```powershell
python -m unittest discover -s tests -t . -v
python -m compileall src tests
```

No command in this README opens the test evaluation gate or runs full project-data training.

## Authorship and citation

> **Human confirmation required:** public authorship order, affiliations, paper title, venue, award information, and citation metadata have not been approved for this repository. Do not infer or cite them from historical files.

## License

Released under the [MIT License](LICENSE).
