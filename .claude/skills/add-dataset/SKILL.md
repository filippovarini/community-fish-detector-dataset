---
name: add-dataset
description: End-to-end workflow for adding a new underwater fish dataset to the community-fish-detector pipeline. Handles research, download testing, script creation, execution, and documentation updates. Use this skill whenever the user shares a link (paper, LinkedIn post, Zenodo, GitHub, LILA, Kaggle, FTP, data portal, etc.) and wants to add it as a dataset, process a new dataset, or integrate new fish/marine data. Even if the user just pastes a URL with minimal context like "add this" or "process this dataset", trigger this skill.
---

# Add Dataset Skill

You are adding a new underwater fish/marine dataset to the community-fish-detector-dataset pipeline. The user has shared a resource: **$ARGUMENTS**

This is a multi-phase workflow. Each phase builds on the previous one. You will use a **tmux session called `cfd-extension`** for all downloads and script execution so the user can attach and monitor progress.

## Phase 1: Research & Extract Metadata

### 1.1 Fetch and analyze the resource

Use WebFetch on the provided link. The link might be a paper, a data portal, a GitHub repo, a LinkedIn post, a Zenodo record — anything. Extract as much metadata as you can from the resource page(s):

**Almost Always extractable from the resource:**
- **Dataset name** and a one-line description
- **Citation** (authors, year, title, DOI if available)
- **License** (if listed; otherwise "N/A")
- **Domain/environment** (coral reef, pelagic, freshwater, brackish, deep-sea, etc.)
- **Geographic location** (if mentioned)
- **Download URL(s)** for images + annotations (direct links, not landing pages)

**Sometimes on the resource page, sometimes only after downloading the data:**
- **Annotation format** (COCO, Pascal VOC, YOLO, CSV, XML, segmentation masks, custom)
- **Compression format** (zip, tar, tar.gz, etc.)
- **Number of images and annotations**
- **Category/species list** (what labels exist in the annotations)
- **Whether images are underwater**

For the second group, record whatever the resource page mentions. Anything missing will be filled in during Phase 2 after downloading and inspecting the actual data. Mark unknown fields as "TBD — will determine after download."

If the initial link is a paper or social media post, look for links to the actual data repository (Zenodo, GitHub releases, LILA, institutional data portal, etc.) and fetch those pages too.

### 1.2 Determine the dataset shortname

Choose a short, lowercase, underscore-separated identifier (e.g., `deep_vision`, `noaa_puget`, `orange_chromide`). Propose it to the user for confirmation.

### 1.3 Check eligibility (preliminary)

Based on what you know so far, STOP and inform the user if any of these clearly apply:
- The dataset is **above-water only** (on-deck cameras are borderline — ask the user)
- There are **no fish-related annotations** — fish, sharks, rays, whales, dolphins, eels, and any fish species are accepted. Crabs, turtles, plastic, debris, humans, coral-only, invertebrates are NOT.
- The dataset is **classification-only** (no bounding boxes or segmentation masks)

If eligibility can't be fully determined from the resource page alone (e.g., the category list isn't published), note this and defer the final check to Phase 2 when you can inspect the actual data.

### 1.4 Present findings to user

Before proceeding, show the user everything you've gathered:
- Dataset name, source, description
- License, domain, location
- Download URL(s) and method
- What you know about annotation format, categories, image count
- What's still TBD and will be confirmed after download
- Any concerns or manual steps required

Ask for confirmation before moving to Phase 2.

## Phase 2: Test Download & Analyze Data

This phase is about verifying the download works and understanding the data structure before writing any script.

### 2.1 Set up tmux session

Create a tmux session for all downloads and script runs:
```bash
tmux new-session -d -s cfd-extension 2>/dev/null || true
```

All download and execution commands in subsequent phases should be sent to this tmux session using:
```bash
tmux send-keys -t cfd-extension '<command>' Enter
```

Tell the user: "I've created a tmux session called `cfd-extension`. You can attach to it with `tmux attach -t cfd-extension` to monitor progress."

### 2.2 Test the download URL

Write a small test script or use curl/wget to test whether the download URL actually works. Run it in the tmux session.

**If the download fails:**
1. Go back to the original resource and look for alternative download links (different mirrors, API endpoints, direct vs redirect URLs)
2. Retry with the alternative URL
3. If it still fails, the dataset likely requires authentication or manual acceptance of terms. Tell the user: "This dataset doesn't seem to have a programmatic download URL. You'll need to download it manually to `<raw_dir>/<shortname>/`. Let me know once the files are there and I'll continue processing."

**If the download succeeds**, proceed to analyze the data.

### 2.3 Analyze the downloaded data

Once you have the data (or a sample of it), examine:

1. **Folder structure**: `ls -R` or `find` to understand how files are organized
2. **Image files**: formats (jpg, png, tif), naming conventions, how many
3. **Annotation files**: open and inspect the format, schema, field names
4. **Metadata files**: any README, CSV, JSON that describes the data
5. **Filename patterns**: do filenames encode location, camera, video, date, or deployment info? This is critical for the split strategy.

### 2.4 Fill in TBD metadata and confirm eligibility

Now that you have the actual data, resolve anything marked "TBD" from Phase 1:
- **Annotation format**: confirmed by inspecting the annotation files
- **Number of images**: count them (`find ... | wc -l` or similar)
- **Number of annotations/bounding boxes**: parse the annotation files
- **Category/species list**: extract the full list from the annotations
- **Whether images are underwater**: sample a few images to confirm

With the full category list now known, finalize the **eligibility check** and **`CATEGORIES_FILTER`**:
- List ALL categories found in the annotations
- KEEP fish-looking animals (fish, shark, whale, dolphin, ray, eel, species names)
- DISCARD non-fish (crab, turtle, coral, starfish, jellyfish, human, debris)
- If ALL categories are fish-relevant, set `CATEGORIES_FILTER = None`

If anything disqualifies the dataset (no fish annotations, above-water only, classification-only), STOP and inform the user.

### 2.5 Determine split strategy

Based on the data analysis, decide the split strategy (in order of preference):
1. **By location/site/deployment** — if filenames or metadata contain location identifiers
2. **By camera/sensor** — if multiple cameras are identifiable
3. **By video ID** — if images are frames extracted from video
4. **By date/time** — if temporal metadata is available
5. **Random split** — last resort only

The goal: all images from one physical location should go into either training or evaluation, never both. This prevents data leakage.

Present the split strategy to the user with your reasoning.

## Phase 3: Create the Dataset Script

Create `datasets/<shortname>.py` following the **exact 4-step pattern** used by all other scripts. Consult [reference.md](reference.md) for the utility API and [examples/](examples/) for complete working examples.

### Script structure

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
    # Import only what you need — see reference.md for full API
)

DATASET_SHORTNAME = "<shortname>"
CATEGORIES_FILTER = [...]  # or None if all categories are fish

settings = Settings()


def download_data(download_path: Path):
    """Download and extract the raw dataset."""
    ...


def main():
    # 1. DOWNLOAD
    raw_download_path = settings.raw_dir / DATASET_SHORTNAME
    raw_download_path.mkdir(parents=True, exist_ok=True)
    download_data(raw_download_path)

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

### Critical rules

- **1-indexed COCO annotations**: Category IDs start at 1, not 0. Use `convert_coco_annotations_from_0_indexed_to_1_indexed` if the source is 0-indexed.
- **Prefix image filenames** with `DATASET_SHORTNAME` using `add_dataset_shortname_prefix_to_image_names` — this prevents filename collisions when datasets are merged.
- **Single "fish" category**: Always compress to a single category using `compress_annotations_to_single_category`.
- **COCO bbox format**: `[x, y, width, height]` (top-left corner + dimensions). Convert from `[xmin, ymin, xmax, ymax]` if needed.
- **Idempotent guards**: Add `if path.exists(): return` checks to avoid re-downloading or re-processing on rerun.
- **Manual download fallback**: If the dataset requires login or term acceptance, have `download_data()` check if files already exist and print instructions if not.

### Annotation format conversions

| Source Format | Conversion Approach |
|---|---|
| COCO JSON | Direct use, just compress categories |
| Pascal VOC XML | `supervision.DetectionDataset.from_pascal_voc()` → export to COCO |
| YOLO TXT | `supervision.DetectionDataset.from_yolo()` → export to COCO |
| CSV with bbox columns | Build COCO dict manually (see deep_vision example) |
| Segmentation masks | Connected components → bounding boxes |
| Custom XML | Parse XML → build COCO dict manually |
| Parquet/HuggingFace | Load with pandas/datasets → build COCO dict |

## Phase 4: Run the Script

Run the dataset script in the tmux session so the user can monitor:

```bash
tmux send-keys -t cfd-extension 'cd <project-root> && python -m datasets.<shortname>' Enter
```

Tell the user it's running and they can monitor with `tmux attach -t cfd-extension`.

**Wait for the script to complete.** Periodically check if the process is still running. Downloads can take a long time — that's expected. Once the script finishes:

1. **Check for errors**: Read any error output. If the script failed, diagnose and fix the issue, then rerun.
2. **Verify outputs exist**:
   - Preview image in `previews/<shortname>_sample_image.png`
   - Train dataset in `<processed_dir>/<shortname>_train/`
   - Val dataset in `<processed_dir>/<shortname>_val/`
3. **Show the preview image** to the user so they can verify the annotations look correct.

If there are errors, fix the script and rerun. Iterate until the pipeline completes successfully.

## Phase 5: Update Documentation

### 5.1 Update README.md

Add an entry for the new dataset under the "Publicly available datasets" section. Follow the existing format exactly:

```markdown
### <Dataset Full Name>

<One-line description>

<Citation: Authors (Year). Title. Journal/Source. DOI/URL>

* Data downloadable via <method> from <source> (<a href="<download_url>">download link</a>)
* License: <license or N/A>
* Metadata raw format: <annotation format>
* Categories/species: <what species/categories>
* Vehicle type: <camera type/deployment method>
* Image information: <number of images>
* Annotation information: <number of bounding boxes>
* Typical animal size in pixels: N/A
* Code to render sample annotated image: <a href="./datasets/<shortname>.py"><shortname>.py</a>

<img src="./previews/<shortname>_sample_image.png" width=700>
```

### 5.2 Update DATASETS.md

Add an entry under "Complete Datasets" (or "Partial Datasets" if manual steps are needed). Update the dataset count in the section header.

```markdown
### <shortname>
- **Source**: [<name>](<url>)
- **Download**: Automatic / Manual - <instructions>
- **Annotations**: <format> → <conversion if any>
- **Category filter**: <categories kept or None>
- **Split**: By <split strategy> (<details>)
```

### 5.3 Update merge_all_datasets.py

Check `datasets/merge_all_datasets.py`. If it has a hardcoded list of datasets, add the new shortname. If it dynamically discovers datasets, no change needed.

## Phase 6: Summary

Present a final summary to the user:

- **Dataset**: shortname, full name, source URL
- **Script**: `datasets/<shortname>.py`
- **Categories**: what was kept/filtered
- **Split strategy**: what was used and why
- **Preview**: path to preview image
- **Docs updated**: README.md, DATASETS.md, merge_all_datasets.py (if applicable)
- **Stats**: number of images, annotations, train/val split counts (read from the output annotations)
- **Manual steps** (if any): what the user still needs to do
