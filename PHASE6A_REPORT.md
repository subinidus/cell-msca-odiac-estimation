# Phase 6A implementation report

## Scope and status

Phase 6A adds a locked final-test evaluation path for the nine validation-frozen artifacts defined by three models (`lightgbm_raw`, `cell_msca_token_no_attention`, `cell_msca_bidirectional`) and train seeds 42, 43, 44. No project test sample was materialized or evaluated while implementing or validating this code. No model was trained, selected, or modified.

The runner is intentionally separate from Phase 3–5 training commands. Its CLI exposes no fit, tuning, early-stopping, checkpoint-selection, split-generation, or hyperparameter command. The only test entry point requires `--allow-final-test`, an absent output path, exact immutable validation-package hashes, and a completed pre-gate audit.

## Frozen protocol

`configs/phase6_locked_test_protocol.json` records:

- validation code SHA `35696c7e8c251e47e12c095f4e9e648a43f97a42`;
- Phase 5A and seed-42 package file SHA-256 values;
- data, persistent split, split-config, and train-only preprocessing hashes;
- the validation-selected raw LightGBM parameters and selection rule;
- nine validation manifest, validation prediction, checkpoint/model, configuration, and Git identities;
- `nonnegative_max_zero_v1` as the unchanged prediction-support policy;
- original-unit MAE/RMSE/R² as primary metrics;
- bias, Spearman, and log-unit MAE/RMSE/R² as secondary metrics;
- three-seed sample standard deviation with `ddof=1`;
- paired complete-cell bootstrap with 10,000 repetitions and seed 3407.

The target remains the `v1_legacy ODIAC-derived zonal mean`. Phase 6A does not introduce a direct-measurement, cell-total, or mass-conservation claim.

## Gate behavior

The gate validates both ZIP file hashes, safe ZIP member paths, all Phase 5A stages, failed-stage count zero, the closed validation test flags, the validation-MAE LightGBM winner, the nine aggregation identities, each run's provenance and config, each model artifact hash, and safe model loading. It verifies that the current clean Git source descends from the frozen validation code. An executable gate now requires the exact verified protocol and immutably records its config/package/data/split/preprocessing/source/model-set identity plus an output-independent deterministic execution ID. Execution revalidates that identity before creating output or materializing test.

A production gate is now bound to the exact in-process `BaselineDataProtocol`, dataset, and persistent-split object identities in addition to its type, provenance, and fingerprint. A separate protocol with identical hashes cannot use that gate, and the gate is not a serializable cross-process authorization token.

A fixed per-environment registry uses atomic exclusive claim creation and the states `claimed`, `materialization_started`, `materialized`, `evaluating`, `finalizing`, `failed`, and `completed`. Production has no registry-path CLI option: Kaggle uses `/kaggle/working/.cell_msca_phase6_final_test_registry`, while local execution uses the corresponding fixed user-home directory. Materialized row/cell/month counts are recorded immediately after access, and every completed model prediction is recorded with artifact paths and SHA-256 values. Failure records preserve whether test was materialized/evaluated, completed and failed model/seed identities, completed artifact SHA-256 values, and the exception. Failed-run recovery remains disabled and fail-closed; separate Kaggle sessions are governed by preserved receipts and the research execution procedure rather than an unsupported claim of centralized global locking.

Final success publication now stages the success manifest, ZIP, and receipt in an isolated execution-ID directory. Their hashes enter the `finalizing` registry state, then the registry transitions atomically to `completed`, and only afterward are public names published. A completed-run publish interruption can resume through SHA- and claimed-output-path-verified publish-only recovery without data materialization, prediction, training, or metric recomputation. Successful and failed recovery attempts are recorded in the registry; altered or redirected staging files are rejected.

The two immutable seed-42 neural manifests predate the explicit `test_evaluation_performed` field. Their narrowly scoped compatibility rule requires the exact package and manifest hashes, `test_subset_materialized=false`, and Phase 5A's completed closed-gate audit. The default rule for every newer artifact remains strict.

After authorization, the runner requires 2,574 unique test cells, 36 months per cell, and 92,664 rows. Each prediction CSV is independently re-read for metric verification. The same stored-versus-recalculated metric check is now executed on every immutable validation prediction during package preflight, including the seed-42 legacy path. It uses documented `rtol=1e-12` and `atol=1e-12`, records actual deltas, retains exact package/manifest/prediction/model hashes and row content, and rejects non-finite metrics. Bootstrap resampling draws unique cells with replacement and carries all 36 monthly rows for each draw. Its interval conditions on the mean of the three frozen-seed predictions and does not include training-seed uncertainty.

## Verification performed

- Full stored suite after the residual review fixes: 147 run; 145 passed, 2 skipped, 0 failed, 0 errors.
  - The skips are the pre-existing optional LightGBM round-trip test in the default environment and an environment-variable-gated external seed-42 ZIP test.
- Phase 6A targeted tests: 27 passed. The added counterexamples cover protocol-less gates, production protocol type enforcement, separate same-provenance protocol instances, output-independent duplicate claims, failure before and immediately after materialization, an Nth-model prediction failure, mandatory failure manifests, disabled failed-run recovery, completed-only publish recovery, altered staging, frozen-model parameter mutation, forbidden training/selection APIs, exact prediction SHA enforcement, and metric tolerance boundaries (`1e-14` accepted, `1e-8` rejected).
- `compileall src tests`: passed.
- `git diff --check`: passed.
- CLI import/help smoke: passed.
- Actual immutable package preflight, without project data or test access: passed.
  - LightGBM 4.6.0 safely loaded the three text models at iterations 4414, 4972, and 3841.
  - PyTorch 2.13.0+cpu loaded all six checkpoints with `weights_only=True`.
  - NumPy 2.5.2 and SciPy 1.18.1 were used in this isolated preflight.
  - loaded artifact count: 9; validation metric preflight count: 9; failed Phase 5A stage count: 0; test materialized: false.
  - maximum stored-versus-recalculated validation metric delta: absolute `2.2737367544323206e-13`, relative `3.0537670151050262e-15`; both satisfy the frozen `rtol=1e-12`, `atol=1e-12` policy.
  - all nine model-state fingerprints were distinct and stable across an eight-row synthetic CPU inference smoke; no optimizer, backward, fit, early-stopping, tuning, or checkpoint-selection path was called.
- Legacy flat `src/*.py` changes relative to the Phase 6A base: none.

The LightGBM/SciPy packages used for the actual safe-load check were placed outside the repository in an isolated temporary dependency directory and removed after verification. The global Python environment was not changed.

## Deferred action

The actual Phase 6 final-test execution remains deliberately pending a separate review and explicit operational decision. Its output will be final-test evidence and must not be used to retune models, change the prediction-support rule, or select a different model.
