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

A fixed per-environment registry uses atomic exclusive claim creation. Production has no registry-path CLI option: Kaggle uses `/kaggle/working/.cell_msca_phase6_final_test_registry`, while local execution uses the corresponding fixed user-home directory. Scientific state (`claimed`, `materialization_started`, `materialized`, `evaluating`, `completed`, `failed`) and publish state (`not_started`, `staging`, `ready`, `publishing`, `published`, `publish_failed`) are independent. A publish error after scientific completion never rewrites scientific state as failed. Materialized row/cell/month counts are recorded immediately after access, and every completed model prediction is recorded with artifact paths and SHA-256 values. Scientific failed-run recovery remains disabled and fail-closed; separate Kaggle sessions are governed by preserved receipts and the research execution procedure rather than an unsupported claim of centralized global locking.

After scientific completion, the success manifest, final ZIP, and receipt are created as the only three entries in `<authoritative-registry>/.staging/<execution_id>/bundle`. The complete set, manifest identity and flags, ZIP/receipt filename-size-SHA contract, public-destination absence, and same-filesystem condition are verified before any public operation. Publication is one atomic directory rename to `<authoritative-registry>/published/<execution_id>`, so no application-created partial public bundle is possible. A publish interruption uses SHA-verified publish-only recovery without data materialization, prediction, training, metric, or bootstrap recomputation. Resolved publish errors remain auditable; altered staging or public artifacts are rejected.

The two immutable seed-42 neural manifests predate the explicit `test_evaluation_performed` field. Their narrowly scoped compatibility rule requires the exact package and manifest hashes, `test_subset_materialized=false`, and Phase 5A's completed closed-gate audit. The default rule for every newer artifact remains strict.

After authorization, the runner requires 2,574 unique test cells, 36 months per cell, and 92,664 rows. Each prediction CSV is independently re-read for metric verification. The same stored-versus-recalculated metric check is now executed on every immutable validation prediction during package preflight, including the seed-42 legacy path. It uses documented `rtol=1e-12` and `atol=1e-12`, records actual deltas, retains exact package/manifest/prediction/model hashes and row content, and rejects non-finite metrics. Bootstrap resampling draws unique cells with replacement and carries all 36 monthly rows for each draw. Its interval conditions on the mean of the three frozen-seed predictions and does not include training-seed uncertainty.

## Transactional publish failure matrix

| Injected boundary | Scientific state | Publish state / outcome | Public result |
|---|---|---|---|
| before first staging artifact | completed | publish_failed | absent |
| after first staging artifact | completed | publish_failed | absent |
| after third staging artifact | completed | publish_failed | absent |
| manifest identity validation | completed | publish_failed | absent |
| ZIP mutation before validation | completed | publish_failed | absent |
| receipt mutation before validation | completed | publish_failed | absent |
| public destination collision | claimed; no test access | not_started | application creates nothing |
| ready-state registry write | completed | publish_failed | absent |
| atomic directory rename | completed | publish_failed, recoverable | absent before recovery; complete after recovery |
| registry write after successful rename | completed | publish_failed, recoverable | complete directory; state repaired without evaluation |
| staged/public mutation during recovery | completed | publish_failed | recovery rejected |

Every application-controlled public transition is one directory `os.replace`, never three file moves. Therefore the application exposes either no `published/<execution_id>` directory or one directory containing exactly the validated manifest, ZIP, and receipt. State-consistency checks reject simultaneous staging/public bundles, unexpected or partial public contents, scientific failure with a public bundle, published state with unresolved errors, and scientific completion with a scientific failure manifest.

## Verification performed

- Full stored suite after the transactional publish fixes: 153 run; 150 passed, 3 skipped, 0 failed, 0 errors.
  - The skips are the optional LightGBM round-trip test in the project environment and two environment-variable-gated immutable seed-42 ZIP tests. The corresponding immutable-package behavior was exercised separately by the actual package preflight below.
- Phase 6A targeted tests: 33 run; 32 passed, 1 external-ZIP test skipped, 0 failed, 0 errors. In addition to the existing gate, training-prohibition, and metric-tolerance counterexamples, the suite injects every staging/validation/ready/rename/post-rename boundary listed above and covers both staged and public recovery mutation.
- `compileall src tests`: passed.
- `git diff --check`: passed.
- CLI import/help smoke: passed.
- Actual immutable package preflight, without project data or test access: passed.
  - Both frozen inputs were verified: `cell-msca-phase5a-full-validation-35696c7e8c25.zip` (`b4fb423e9d545a7fab654e6049416898bdc31ea3a64c6c7e414ad727ec74e7a3`) and `cell-msca-v1-legacy-validation-034712a.zip` (`f40aa8db3cff1a56933a69abe9397f43e8df9daa642319848c928cea4ed98555`).
  - LightGBM 4.7.0 safely loaded the three text models at iterations 4414, 4972, and 3841.
  - PyTorch 2.13.0+cpu loaded all six checkpoints with `weights_only=True`.
  - NumPy 2.5.2 and SciPy 1.18.1 were used in this isolated preflight.
  - loaded artifact count: 9; validation metric preflight count: 9; failed Phase 5A stage count: 0; test materialized: false.
  - maximum stored-versus-recalculated validation metric delta: absolute `2.2737367544323206e-13`, relative `3.0537670151050262e-15`; both satisfy the frozen `rtol=1e-12`, `atol=1e-12` policy.
  - all nine model-state fingerprints were distinct and stable across an eight-row synthetic CPU inference smoke; no optimizer, backward, fit, early-stopping, tuning, or checkpoint-selection path was called.
- Legacy flat `src/*.py` changes relative to the Phase 6A base: none.

The LightGBM/SciPy packages used for the actual safe-load check were placed outside the repository in an isolated temporary dependency directory and removed after verification. The global Python environment was not changed.

## Deferred action

The actual Phase 6 final-test execution remains deliberately pending a separate review and explicit operational decision. Its output will be final-test evidence and must not be used to retune models, change the prediction-support rule, or select a different model.
