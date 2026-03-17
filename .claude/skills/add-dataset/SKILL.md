---
name: add-dataset
description: Add a new underwater fish dataset to the pipeline. Use when the user shares a link to a dataset (Zenodo, GitHub, LILA, Kaggle, FTP, etc.) and wants to integrate it into the community-fish-detector-dataset project.
argument-hint: <dataset-url-or-resource>
disable-model-invocation: true
allowed-tools: Read, Write, Edit, Glob, Grep, Bash, Agent, WebFetch
---

# Add Dataset Skill

You are adding a new underwater fish/marine dataset to the community-fish-detector-dataset pipeline. The user has provided a dataset resource: **$ARGUMENTS**

## Phase 1: Research the Dataset

Fetch and analyze the dataset resource to gather metadata:

1. **Fetch the resource page** using WebFetch to extract:
   - Dataset name and description
   - Download URL(s) for the data files (images + annotations)
   - Annotation format (COCO, Pascal VOC, YOLO, CSV, XML, segmentation masks, etc.)
   - File format and compression (zip, tar, etc.)
   - Number of images and annotations (if listed)
   - License information
   - Any papers or citations
   - Whether images are underwater or above-water

2. **Determine the dataset shortname**: a short, lowercase, underscore-separated identifier (e.g., `deep_vision`, `noaa_puget`). Ask the user to confirm.

3. **Check eligibility** — STOP and inform the user if:
   - The dataset contains **above-water images** (only underwater datasets are accepted)
   - The dataset has **no fish-related annotations** (fish, sharks, rays, whales, eels, marine fish species are accepted; turtles, plastic, debris, humans, coral-only are NOT)
   - The dataset has no bounding box or segmentation annotations (classification-only datasets cannot be used)

4. **Identify fish-relevant categories**: List ALL annotation categories from the dataset. Determine which ones to KEEP (fish, sharks, rays, whales, eels, and any fish species) and which to DISCARD (turtles, plastic, debris, humans, invertebrates, coral, etc.). This becomes the `CATEGORIES_FILTER` list. If ALL categories are fish-related, set `CATEGORIES_FILTER = None`.

5. **Determine the split strategy** (in order of preference):
   - By location/site/deployment if identifiable from filenames or metadata
   - By camera/sensor if multiple cameras
   - By video ID if frames extracted from video
   - By date/time if temporal metadata available
   - Random split as last resort

6. **Present findings to the user** and ask for confirmation before proceeding. Include:
   - Dataset name, source, description
   - Download method and URLs
   - Annotation format and conversion needed
   - Categories to keep vs discard
   - Proposed split strategy
   - Any manual steps required

## Phase 2: Create the Dataset Script

Create `datasets/<shortname>.py` following the **exact 4-step pattern** used by all other scripts. For detailed utility API reference and code examples, see [reference.md](reference.md) and [examples/](examples/).

### Script Structure

```python
"""
<Dataset Name>
Source: <source URL>
Split logic: <description of split strategy>
Categories kept: <which categories are kept>
"""

import json
# ... other imports as needed
from pathlib import Path

from datasets.settings import Settings
from datasets.utils import (
    # Import only what you need from:
    # download_and_extract, download_file, extract_downloaded_file, CompressionType,
    # compress_annotations_to_single_category,
    # convert_coco_annotations_from_0_indexed_to_1_indexed,
    # add_dataset_shortname_prefix_to_image_names,
    # copy_images_to_processing,
    # split_coco_dataset_into_train_validation,
    # get_train_images_with_random_splitting,
    # save_preview_image,
)


DATASET_SHORTNAME = "<shortname>"
CATEGORIES_FILTER = [...]  # or None if all categories are fish
# DATA_URL = "<url>"  # if automatic download

settings = Settings()


def download_data(download_path: Path):
    """Download and extract the raw dataset."""
    # Use download_and_extract() for simple URL downloads
    # Or implement custom download logic


def create_coco_dataset(...):
    """Convert raw annotations to COCO format (if not already COCO)."""
    # Only needed if source format is not COCO


def get_list_of_<groups>_to_include_in_train_set(image_folder: Path) -> list[str]:
    """Determine train/val split based on <split strategy>."""
    # Return list of group identifiers for the training set


def main():
    # 1. DOWNLOAD
    raw_download_path = settings.raw_dir / DATASET_SHORTNAME
    raw_download_path.mkdir(parents=True, exist_ok=True)
    # ... download logic

    # 2. PROCESS
    processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
    processing_dir.mkdir(parents=True, exist_ok=True)
    # ... convert to COCO if needed
    # ... compress_annotations_to_single_category()
    # ... add_dataset_shortname_prefix_to_image_names()

    # 3. PREVIEW
    # save_preview_image(images_path, annotations_path, DATASET_SHORTNAME)

    # 4. SPLIT
    # ... determine train set
    # split_coco_dataset_into_train_validation(...)


if __name__ == "__main__":
    main()
```

### Critical Rules for the Script

- **COCO annotations must be 1-indexed** (category IDs start at 1, not 0). Use `convert_coco_annotations_from_0_indexed_to_1_indexed` if source is 0-indexed.
- **Image filenames must be prefixed** with `DATASET_SHORTNAME` using `add_dataset_shortname_prefix_to_image_names` to avoid collisions when merging.
- **Single "fish" category**: Always compress to single category using `compress_annotations_to_single_category`.
- **COCO bbox format**: `[x, y, width, height]` (top-left corner + dimensions). Convert from other formats if needed (e.g., `[xmin, ymin, xmax, ymax]` needs `width = xmax - xmin`).
- **Check for existing files**: Add guards like `if path.exists(): return` to avoid re-downloading or re-processing.
- If the dataset requires **manual download** (e.g., requires login, acceptance of terms), implement the processing steps but have `download_data()` print instructions and check if files exist.

### Annotation Format Conversions

Depending on the source format, you may need to convert annotations:

| Source Format | Conversion Approach |
|---|---|
| COCO JSON | Direct use, just compress categories |
| Pascal VOC XML | Use `supervision.DetectionDataset.from_pascal_voc()` then export to COCO |
| YOLO TXT | Use `supervision.DetectionDataset.from_yolo()` then export to COCO |
| CSV with bbox columns | Build COCO dict manually (see deep_vision example) |
| Segmentation masks | Use connected components to extract bounding boxes |
| Custom XML | Parse XML and build COCO dict manually |

## Phase 3: Update DATASETS.md

Add an entry for the new dataset to `DATASETS.md` under the appropriate section ("Complete Datasets" if fully automatic, "Partial Datasets" if manual steps needed). Follow the existing format:

```markdown
### <shortname>
- **Source**: [<name>](<url>)
- **Download**: Automatic / Manual - <instructions>
- **Annotations**: <format> -> <conversion if any>
- **Category filter**: <categories kept or None>
- **Split**: By <split strategy> (<details>)
- **Special**: <any special requirements> (only if applicable)
```

Also update the dataset count in the section header.

## Phase 4: Update merge_all_datasets.py

Check `datasets/merge_all_datasets.py` — if it has a hardcoded list of datasets, add the new shortname to it. If it dynamically discovers datasets, no change needed.

## Phase 5: Summary

After creating all files, present a summary:
- Dataset shortname and source
- Script location: `datasets/<shortname>.py`
- Categories kept/filtered
- Split strategy used
- Any manual steps the user needs to take (e.g., download files manually, install dependencies)
- How to run: `python -m datasets.<shortname>`
