# Community Fish Detector Dataset

Dataset compilation pipeline for the [community-fish-detector](https://github.com/WildHackers/community-fish-detector) — a model that detects a single label ("fish") in underwater images. This repo aggregates multiple publicly available datasets into a unified COCO-format dataset.

## Project Structure

```
datasets/                          # Unified dataset processing pipeline
  __init__.py
  settings.py                     # Global settings (paths, split ratio, COCO config)
  merge_all_datasets.py           # Merges all processed datasets
  utils/
    __init__.py                   # Re-exports all utilities
    download.py                   # download_file, extract_downloaded_file, download_and_extract
    visualization.py              # visualize_supervision_dataset, save_preview_image
    coco.py                       # compress_annotations_to_single_category, convert_0_to_1_indexed
    images.py                     # add_dataset_shortname_prefix, remove_prefix, copy_images_to_processing
    split.py                      # split_coco_dataset_into_train_validation, get_train_images_with_random_splitting
  <dataset>.py                    # Per-dataset unified script (download + process + preview + split)

previews/                          # Sample annotated images output directory
DATASETS.md                        # Per-dataset processing details
```

## Pipeline Overview

Each `datasets/<dataset>.py` script follows a single unified 4-step pattern:

1. **Download** — downloads/extracts the raw dataset
2. **Process** — converts annotations to COCO format, compresses categories to single "fish" category (id=1), prefixes image filenames with dataset shortname
3. **Preview** — saves a sample annotated image to `previews/`
4. **Split** — splits into train/val by location/source identifier when possible

Final output: COCO-format datasets with one category ("fish"), merged via `datasets/merge_all_datasets.py`.

## Key Settings

Take them from `datasets/settings.py`


## Datasets

**Complete datasets (17)**: brackish, coralscapes, deep_vision, deepfish, f4k, fathomnet, fishclef, kakadu, marine_detect, mit_river_herring, noaa_puget, orange_chromide, project_natick, roboflow_fish, torsi, viame_fishtrack, zebrafish.

See [DATASETS.md](DATASETS.md) for per-dataset processing details including source URLs, download instructions, annotation formats, category filters, and split logic.

## Development

### Dependencies
```
pip install -r requirements.txt
```

Key libraries: supervision, opencv-python, pandas, numpy, fathomnet, kagglehub, tqdm, scikit-learn, roboflow.

### Adding a New Dataset
1. Create `datasets/<name>.py` following the 4-step pattern (download, process, preview, split)
2. Define `DATASET_SHORTNAME`, `CATEGORIES_FILTER`, `download_data()`, and `main()`
3. Use shared utilities from `datasets/utils/`
4. Document the dataset in `DATASETS.md`

### Conventions
- Each dataset script defines `DATASET_SHORTNAME` and `CATEGORIES_FILTER` at module level
- All scripts use shared utilities from `datasets/utils/`
- COCO annotations must be 1-indexed
- Image filenames are prefixed with dataset shortname to avoid collisions when merging
- Train/val split should be by location/deployment/camera/video when possible, not by random image selection
