# data-preprocessing

This README documents only the data preprocessing code in `src/preprocessing`.
The pipeline prepares monthly grid-based input feature tables, continuous ODIAC
regression labels, merged training CSV files, and compressed spatial tensor
samples for carbon-emission regression.

The current project predicts a continuous ODIAC-derived carbon-emission value
rather than a discrete hotspot class. Some preprocessing functions may still
generate legacy class labels and quantile thresholds for compatibility with
earlier pipeline outputs.

## Preprocessing Modules

The preprocessing package is organized by pipeline stage:

| File | Purpose |
| --- | --- |
| `grid.py` | Creates a regular spatial grid from the configured region bounding box. |
| `gee_extract.py` | Extracts raster features from Google Earth Engine by grid cell. |
| `vector_features.py` | Extracts vector power plant features and assigns them to grid cells. |
| `cleaning.py` | Fills missing raster feature values using the configured fill method. |
| `raster_label.py` | Extracts monthly ODIAC label values from local GeoTIFF files. |
| `labeling.py` | Legacy utility that can generate quantile-based class labels for backward compatibility. These class labels are not used by the current regression model. |
| `merge.py` | Merges cleaned features and labels into final monthly datasets. |
| `tensor_builder.py` | Converts final monthly CSV files into `.npz` spatial tensor samples. |

The pipeline entry point is:

```bash
python scripts/run_preprocessing.py --config configs/delhi_2022_2024.yaml
```

## Pipeline Flow

1. Create the grid

   `create_grid()` builds a rectangular grid from the configured region bounds.
   The region is projected into the configured grid CRS, split into fixed-size
   cells, clipped to the bounding box, and saved as both GeoJSON and CSV.

   Main outputs:

   ```text
   data/grid/delhi_ncr_1km_grid.geojson
   data/grid/delhi_ncr_1km_grid.csv
   ```

2. Extract raster features from Google Earth Engine

   `extract_gee_feature()` reads the grid GeoJSON and extracts configured image
   features for each month. Monthly features are averaged over the month, while
   yearly features use the requested year or a configured fallback year.

   Supported raster feature types:

   - `image`
   - `landcover_fraction`

   Supported reducers:

   - `mean`
   - `sum`

   Batch CSV files are written during extraction so existing batches can be
   reused.

3. Extract vector power plant features

   `extract_vector_feature()` currently supports the `power_plants` feature.
   It loads the WRI Global Power Plant Database from Earth Engine, filters it to
   the grid bounding box, spatially joins plants to grid cells, and produces:

   - total power plant count per grid cell
   - total fossil-fuel power capacity in MW per grid cell

4. Clean feature CSV files

   `clean_feature()` fills missing raster feature values after extraction.
   The fill method is configured per feature and supports:

   - `mean`
   - `median`
   - `zero`

   If all values are missing and the fill method is not `zero`, the pipeline
   raises an error instead of silently producing invalid data.

5. Extract ODIAC raster labels

   `extract_odiac_label()` reads the monthly ODIAC GeoTIFF file, reprojects the
   grid if needed, and computes zonal statistics for each grid cell.

   It stores:

   - monthly mean ODIAC value as the regression label
   - log-transformed label
   - ODIAC minimum and maximum values for inspection

6. Preserve optional legacy class labels

   `label_reg` and `label_reg_log` are the active learning targets for the
   current regression project. The existing preprocessing pipeline may still run
   the class-labeling stage and generate quantile-based `label_cls` values for
   backward compatibility.

   These class labels are not used as model input features, training targets,
   loss-function targets, or current evaluation targets. The current pipeline
   does not require class interpretation, class balance analysis, ROC-AUC,
   PR-AUC, or classification accuracy.

   Supported quantile scopes:

   - `per_month`: thresholds are computed independently for each month
   - `global_range`: one threshold set is computed across all configured months

7. Build final monthly datasets

   `build_final_dataset()` merges all cleaned feature tables with ODIAC
   regression labels and any optional legacy label columns on `grid_id` and
   `date`. It validates required columns, checks duplicate grid-date pairs,
   rejects missing values in major columns, and saves one final CSV per month.
   The final dataset must contain the continuous regression target.

8. Build combined dataset

   `build_combined_dataset()` concatenates all monthly final CSV files into one
   multi-month dataset and validates that grid-date pairs remain unique.

9. Build tensor samples

   `build_tensor()` converts each final monthly CSV into a compressed `.npz`
   tensor file. Feature channels are split by the configured stream:

   - `stream_a`
   - `stream_b`

   Each tensor file contains:

   ```text
   stream_a
   stream_b
   label_cls
   label_reg
   mask
   stream_a_features
   stream_b_features
   date
   region
   height
   width
   ```

   `stream_a` and `stream_b` remain unchanged as the model inputs. `label_reg`,
   or its log-transformed equivalent `label_reg_log` when available in the
   final CSV, is the current training target. `label_cls`, when present, is
   legacy metadata and is ignored by the regression model.

   `build_tensor_index()` writes a CSV index that maps each month to its tensor
   file and final dataset file.

## Current Training Target

The current model is trained as a regression model.

- Inputs: `stream_a` and `stream_b`
- Target: `label_reg_log` when available, otherwise `label_reg`
- Output: one continuous emission prediction per sample
- Classification labels: not used for training or evaluation

Legacy `label_cls` columns may remain in intermediate CSV or tensor files so
that existing preprocessing outputs remain compatible with earlier versions of
the project.

## Configuration-Driven Execution

The script `scripts/run_preprocessing.py` loads a YAML config and runs stages
based on the `run` flags:

```yaml
run:
  create_grid: true
  extract_features: true
  clean_features: true
  extract_label: true
  create_label_cls: true
  build_final_dataset: true
  build_tensor: true
  build_combined_dataset: true
  build_tensor_index: true
```

The `create_label_cls: true` flag represents a legacy compatibility stage in
the current code path. It does not mean that classification is used for current
model training.

The config also controls:

- project name and Google Earth Engine project ID
- region bounds, CRS settings, and grid cell size
- monthly date range
- feature collections, bands, reducers, streams, scales, and fill methods
- ODIAC file path pattern
- regression label column names and optional legacy class-label settings
- output directories

## Expected Inputs

Google Earth Engine authentication must be available before running stages that
extract GEE features.

Monthly ODIAC GeoTIFF files must exist locally according to the configured path
pattern. The current path helper supports either:

```yaml
paths:
  odiactif_pattern: data/raw/odiac/odiac2025_1km_excl_intl_{yy}{mm}.tif
```

or:

```yaml
paths:
  odiac_tif_pattern: data/raw/odiac/odiac2025_1km_excl_intl_{yy}{mm}.tif
```

## Main Outputs

The preprocessing pipeline writes intermediate and processed files under the
configured directories.

Typical intermediate outputs:

```text
data/interim/<stream>_<feature>_YYYY_MM.csv
data/interim/<feature>_batches_YYYY_MM/<feature>_batch_000.csv
```

Typical processed outputs:

```text
data/processed/<stream>_<feature>_YYYY_MM_clean.csv
data/processed/odiac_label_YYYY_MM.csv
data/processed/odiac_label_YYYY_MM_cls.csv                  # legacy compatibility output
data/processed/final_ms_mca_dataset_YYYY_MM.csv
data/processed/final_ms_mca_dataset_<start_year>_<end_year>.csv
data/processed/ms_mca_tensor_<region>_YYYY_MM.npz
data/processed/tensor_index_<start_year>_<end_year>.csv
data/processed/label_thresholds_<start_year>_<end_year>_global.csv  # legacy compatibility output
```

The primary processed outputs for current regression training are the final
monthly dataset CSV files, the combined multi-month dataset, and tensor files
that contain the continuous ODIAC regression target.

## Validation Behavior

The preprocessing code performs validation throughout the pipeline:

- required files must exist before they are read
- required columns must be present in CSV and GeoDataFrame inputs
- unsupported feature types, reducers, temporal resolutions, and fill methods
  raise errors
- all-missing features cannot be filled with `mean` or `median`
- ODIAC regression labels cannot contain missing values before final dataset or
  tensor construction
- legacy class-label generation also validates that labels are present before
  assigning quantile classes
- final monthly datasets cannot contain duplicate `grid_id` and `date` pairs
- final monthly datasets cannot contain missing values in required columns
