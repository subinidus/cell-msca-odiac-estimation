# Phase 1 Implementation Report

## Scope implemented

- Shared `log1p(y / scale)` target transform with negative-target validation
- Shared median and Duan-smearing inverse transform paths
- Original-unit and log-space metric namespaces
- Original-unit and log-space cell-cluster bootstrap APIs and outputs
- Prediction CSV metric recalculation and JSON consistency verification
- Synthetic unit and regression tests for the Phase 1 completion gates

## Files changed

- `src/cell_msca/__init__.py`
- `src/cell_msca/target.py`
- `src/cell_msca/metrics.py`
- `src/cell_msca/evaluate.py`
- `tests/__init__.py`
- `tests/test_target.py`
- `tests/test_metrics.py`
- `tests/test_prediction_roundtrip.py`
- `PHASE1_REPORT.md`

The pre-existing flat `src/` modules were copied from the supplied baseline
archive and were not edited.

## Verification status

Test command:

```powershell
python -m unittest discover -s tests -t . -v
```

The command was run with the bundled workspace Python runtime and NumPy 2.3.5.
The default system Python was not used because it did not contain the project's
NumPy dependency.

Result: **12 tests run, 12 passed**.

| Completion gate | Test result |
|---|---|
| Perfect predictions give MAE=0, RMSE=0, and R2=1 in both spaces | Passed |
| `pred_log=0`, smearing factor `S=2` gives corrected prediction `1` | Passed |
| Original-unit bootstrap receives original-unit arrays, not log arrays | Passed |
| Saved prediction CSV reproduces JSON metrics exactly | Passed |
| Paper-facing metric output has no hotspot field | Passed |

Additional checks:

- `compileall` passed for `src/cell_msca` and `tests`.
- All 11 pre-existing flat `src/*.py` files match the supplied baseline ZIP by
  SHA-256; no dataset, split, model, or training source was changed.
- No result, checkpoint, `.pth`, or `.pt` artifact was created.
- No training command was run.

## Unresolved items

- Selection of median versus Duan-smearing inverse remains a validation-stage
  decision and must be recorded before opening the test split.
- The physical target unit and the value of `scale` remain dependent on the
  ODIAC source-unit verification described in the implementation specification.
- Dataset, split, model, and training integration belongs to later phases and
  is intentionally not implemented here.
- The supplied environment has no `ruff` executable, so Ruff linting was not
  run. Syntax compilation and all unit tests passed.
