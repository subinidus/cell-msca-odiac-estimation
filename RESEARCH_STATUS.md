# Research Status

Status date: 2026-08-23
Active branch scope: Phase 4 architecture and synthetic validation only

## Current research statement

The project studies same-period, cell-level estimation of a `v1_legacy ODIAC-derived zonal mean` for Delhi NCR from seven same-cell proxy features. ODIAC is an inventory-based spatial proxy, not a direct satellite CO2 measurement, physical ground truth, or independent field monitor. The exact ODIAC release and physical interpretation of the archived target remain unresolved.

No current Cell-MSCA project-data performance result exists. The earlier 16x16 experiment is preserved only as historical material under `docs/legacy/`.

## Completed phases

| Phase | Completed scope |
|---|---|
| 1 | Shared target transform and inverse, median and train-residual-only Duan modes, original/log metrics, separate cell-cluster bootstrap, prediction CSV reproduction |
| 2 | One-cell dataset, fixed feature metadata and order, persistent cell-fixed split, split/data/config hashes, separated split and train seeds, train-only runtime preprocessing |
| 2.1 | Safe metadata-required split loading, preprocessing provenance binding, actual split ratios, development buffered-split distance metadata |
| 3 | Train mean, raw/log1p/Tweedie LightGBM, Concat-MLP, shared evaluator and result schema, validation-only tuning, frozen test gate |
| 3.5 | Primary-source literature audit, project-file audit, claim-evidence matrix, verified bibliography |

## Verified implementation components

- Input samples expose pollution/environment `[3]` and socio-infrastructure `[4]` arrays, targets, cell ID, and month ID without neighbours or coordinates.
- Feature order is fixed as `no2_mean`, `so2_mean`, `co_mean`, then `nightlight_mean`, `urban_fraction`, `power_plant_count`, `fossil_capacity_mw`.
- The primary split keeps every month of a cell in one subset and is reused through a persistent checksum.
- Runtime normalization and imputation statistics are fitted on train cells and are bound to data and split hashes.
- Original-unit MAE, RMSE, R², and Bias are headline metrics; log metrics and Spearman are secondary.
- Log-model checkpoints use median-inverse validation original-unit MAE. Median versus Duan is compared only after checkpoint selection, and Duan uses train residuals only.
- Tuning materializes train and validation only. Test access requires a frozen selection file with matching provenance hashes.
- The legacy flat `src/*.py` implementation is isolated from the active `src/cell_msca/` package.

## Unresolved evidence blockers

1. The exact 36 ODIAC source filenames, release identifier, checksums, and raster headers are unavailable.
2. Source CRS, affine transform, resolution, nodata, unit, time semantics, and carbon-versus-CO2 convention are not confirmed from the generating artifacts.
3. It is not established whether `v1_legacy` values are totals, densities, or another native-raster quantity after zonal averaging.
4. Boundary-cell geometry and a physically justified mass- or density-preserving `v2_corrected` operation are unresolved.
5. Legacy feature extraction filled missing values with month-wide statistics before the model split; raw pre-imputation features must be recovered.
6. Nightlight directly overlaps a documented ODIAC construction proxy; exact power-source dataset overlap remains unresolved.
7. The primary scientific estimand—conditional median, conditional mean, or regional total—requires an explicit human decision.
8. Final block and buffer distances require empirical spatial autocorrelation and anisotropy analysis.
9. Independent Delhi inventory or top-down external validation data have not been secured.
10. Public authorship, affiliation, venue, award, and citation metadata require human confirmation.

## Work allowed before data recovery

- Implement and test the feature-specific numerical tokenizer.
- Implement token/no-attention, forward `I <- P`, reverse `P <- I`, and bidirectional architectures.
- Run synthetic CPU/CUDA compatibility, forward, backward, determinism, batch-isolation, checkpoint, and parameter-count tests.
- Extend reusable neural device, seed, optimizer-group, checkpoint, and provenance utilities without opening the test gate.
- Design—but do not execute on project data—the mandatory nightlight and power-feature overlap ablations.
- Specify target-manifest, corrected-aggregation, and empirical spatial-validation requirements.

## Work prohibited from becoming paper-facing results

- Any number from the historical 16x16 experiment.
- Any architecture smoke result or synthetic loss value.
- Any `v1_legacy` result described as a confirmed physical total or direct-emissions accuracy.
- Any result selected, tuned, or interpreted using the test subset.
- Any hotspot, policy, causal-attention, source-attribution, strict-spatial-independence, or forecasting claim under the current protocol.
- Any `v2_corrected` result created without recovered source metadata and a validated aggregation branch.

## Conditions before Phase 4 project-data training

All conditions below are mandatory:

1. Recover and checksum the exact 36 source rasters and record the complete source/derived manifest required by `IMPLEMENTATION_SPEC.md`.
2. Resolve the source variable and unit, then approve and test the appropriate mass- or density-preserving target equation and boundary policy.
3. Create a new immutable corrected data version from raw pre-imputation features; never overwrite `v1_legacy`.
4. Verify feature licenses, image revisions, QA/cloud filters, dates, and extraction logs.
5. Decide and document the primary estimand and corresponding inverse/reporting rule.
6. Predefine the empirical spatial-autocorrelation diagnostic and candidate block/buffer protocol.
7. Freeze feature-overlap ablations, architecture capacity controls, optimizer, loss candidates, and train-seed schedule.
8. Pass every synthetic architecture, checkpoint, provenance, test-gate, and legacy-hash regression test.

Until these conditions are satisfied, only architecture engineering and synthetic validation are authorized.

## Conditions before the first test-set evaluation

1. Freeze a verified data version and its SHA-256 manifest.
2. Freeze the persistent split and split-config hashes without changing `split_seed`.
3. Fit and freeze preprocessing on train cells only; bind its hash to the exact data and split.
4. Complete validation-only model, loss, checkpoint, inverse-mode, and hyperparameter selection.
5. Compute any Duan factor from train residuals only and freeze it before test access.
6. Record all train seeds, runtime versions, configuration hashes, parameter counts, and Git commit SHA.
7. Create and independently validate the frozen selection authorization file.
8. Confirm no test subset was materialized during tuning and obtain explicit human approval to open the gate.
9. Evaluate the test subset once under the frozen protocol and preserve predictions for metric reproduction and cell-cluster uncertainty.

## Human decisions still required

- exact ODIAC release and physical target interpretation;
- carbon or CO2 reporting convention;
- boundary-cell reporting convention;
- mean versus median scientific estimand;
- final spatial robustness distances and feasibility threshold;
- legacy-result appendix inclusion;
- authorship, affiliation, title, venue, award, and citation metadata.
