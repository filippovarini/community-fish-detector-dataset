# Community Fish Detector Dataset

Dataset compilation pipeline for the [community-fish-detector](https://github.com/WildHackers/community-fish-detector) — a model that detects a single label ("fish") in underwater images. This repo aggregates multiple publicly available datasets into a unified COCO-format dataset.

## Project Structure

```
data_preview/           # Phase 1: dataset preview scripts
  visualize_<dataset>.py   # Download, process annotations, visualize a sample image with bboxes
  utils.py                 # Shared visualization utilities
  *.ipynb                  # Notebook-based previews for some datasets

aggregation_of_final_dataset/  # Phase 2: dataset processing pipeline
  <dataset>.py             # Per-dataset processing script
  settings.py              # Global settings (paths, split ratio, COCO config)
  utils.py                 # Shared utilities (category compression, splitting, renaming)
  merge_all_datasets_into_one.py  # Merges all processed datasets
  upload_to_roboflow.py    # Upload final dataset to Roboflow
```

## Pipeline Overview

### Phase 1: Data Preview (`data_preview/`)
Each `visualize_<dataset>.py` script:
1. Downloads the dataset
2. Processes annotations into COCO format
3. Visualizes a random image with bounding boxes overlaid

### Phase 2: Data Processing (`aggregation_of_final_dataset/`)
Each `<dataset>.py` script:
1. Downloads and cleans the data
2. Filters categories to keep only fish-related ones
3. Compresses all fish categories into a single "fish" category (id=1)
4. Converts annotations to 1-indexed COCO format
5. Prefixes image filenames with dataset shortname (e.g., `brackish_image001.jpg`)
6. Splits into train/val — split is done by location/source identifier (not random image split) when possible. The split logic varies per dataset (by deployment site, camera, video, or random when no grouping is available)

Final output: COCO-format datasets with one category ("fish"), merged via `merge_all_datasets_into_one.py`.

## Key Settings (`aggregation_of_final_dataset/settings.py`)

- **Data paths**: raw → `/mnt/data/dev/fish-datasets/data/raw`, processed → `.../final`, intermediate → `.../processing`
- **Train/val split ratio**: 0.2 (20% validation)
- **COCO category**: single category `{"id": 1, "name": "fish"}`
- **Images folder**: `JPEGImages/`
- **Annotations file**: `annotations_coco.json`

## Datasets

Preview scripts exist for: brackish, deep_vision, deepfish, fathomnet, fishclef, mit_river_herring, noaa_puget, project_natick, roboflow_fish, salmon_computer_vision, torsi, viame_fishtrack, zebrafish, coralscapes, f4k, kakadu, marine_detect (fishinv & megafauna), vmat, ozfish, fishnet, affine.

Aggregation scripts exist for: brackish, deep_vision, deepfish, fathomnet, fishclef, mit_river_herring, noaa_puget, project_natick, roboflow_fish, viame_fishtrack, zebrafish.

## Development

### Dependencies
```
pip install -r requirements.txt
```

Key libraries: supervision, opencv-python, pandas, numpy, fathomnet, kagglehub, tqdm, scikit-learn, roboflow.

### Adding a New Dataset
1. Create `data_preview/visualize_<name>.py` — implement download, annotation parsing, and sample visualization
2. Create `aggregation_of_final_dataset/<name>.py` — implement category filtering, category compression, train/val splitting (by location/source when possible), and COCO output
3. Import the new dataset in `merge_all_datasets_into_one.py`

### Conventions
- Dataset preview scripts define `DATASET_SHORTNAME` and `download_data()` which are imported by the aggregation scripts
- All aggregation scripts use shared utilities from `aggregation_of_final_dataset/utils.py`
- COCO annotations must be 1-indexed
- Image filenames are prefixed with dataset shortname to avoid collisions when merging
- Train/val split should be by location/deployment/camera/video when possible, not by random image selection
