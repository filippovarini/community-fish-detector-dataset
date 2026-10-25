# Dataset compilation

How the Community Fish Detection dataset is built from the individual datasets, in three steps. Run all commands from the repo root.

```
1. Process each dataset   python -m datasets.<name>                   -> data/final/<name>_{train,val}/
2. Deduplicate            python -m quality_checks.deduplicate        -> data/quality/dedup/pairs.csv, report.md
                          python -m quality_checks.select_duplicates  -> data/quality/dedup/images_to_remove.csv
3. Merge                  python -m datasets.merge_all_datasets       -> data/community_fish_detector/
```

## 1. Dataset processing

Each `datasets/<name>.py` downloads one dataset, maps its classes to fish / non-fish / discard, prefixes the image file names with the dataset shortname, and splits it into train and val (by location, camera or video where possible). See [CLAUDE.md](CLAUDE.md) for the class definitions and [DATASETS.md](DATASETS.md) for the details of each dataset.

## 2. Deduplication

Some datasets contain images from other datasets. `quality_checks/deduplicate.py` (adapted from GerardCB's [deduplicate.py](https://github.com/GerardCB/community-fish-detector-dataset/blob/gc/dedup/datasets/deduplicate.py)) finds them:

- Each image gets an [SSCD](https://github.com/facebookresearch/sscd-copy-detection) copy-detection descriptor, which stays nearly the same when a photo is resized, re-encoded, slightly cropped, stretched or overlaid with text. Each image is compared with its nearest neighbours in the other datasets, in the other split of its dataset, and in its own split, by cosine similarity.
- A pair with similarity >= 0.9 is a duplicate. Unrelated photos stay below ~0.4, and neighbouring frames of a video reach 0.75-0.9. Files with the same SHA-1 are exact copies.
- It needs a GPU and `torch` (`pip install "torch>=2.1"`). Most of the time goes into reading the images (~80 images/s from network storage). Descriptors are cached per dataset split in `data/quality/dedup/descriptors/`, so a rerun only processes new or changed images.
- Outputs: `report.md` (summary), `pairs.csv` (all candidate pairs), `clusters.json`, `near_identical_frames.csv`, and side-by-side review sheets in `review/` with `--review 24`.

`quality_checks/select_duplicates.py` then applies the decisions below and writes `images_to_remove.csv`. Nothing is deleted from `data/final`. Each rule removes every image of the first dataset that has a match >= 0.9 in the second. Any match counts, not only the best one, so neighbouring frames of a copied video frame are removed too.

| Remove from | Matched in | Images | Reason |
|---|---|---|---|
| fathomnet | viame_fishtrack | 202,163 | NOAA SEFSC SEAMAP survey footage is in both (`gfisher/` URLs in FathomNet). VIAME keeps it, with its sequence info and split by camera drop |
| roboflow_fish | marine_detect | 573 | ~85% of roboflow_fish is in marine_detect |
| uiis10k | trashcan | 161 | uiis10k reuses TrashCan images |
| ruod_r | marine_detect | 34 | Shared web images |
| uiis10k | ruod_r | 6 | Shared web images |

Total: 202,937 images removed (run of 2026-10-24, 1,322,009 images in 40 datasets).

Not removed:
- **Same scene in train and val of one dataset**, mostly in fixed-camera datasets (brackish, kakadu, project_natick, mit_river_herring). These are different frames of the same scene, not copies, and the split is already by camera or video where possible. Val scores on these datasets will be optimistic.
- **FathomNet's train/val split** is random by image, so frames of the same dive can be in both splits. The SEAMAP removal fixes most of it; the rest stays for now.
- **Near-identical frames inside one split** (repeated video frames, ~33k byte-identical empty frames in eyesea). Listed in `near_identical_frames.csv` for subsampling in training.
- **f4k and fishclef** share the Fish4Knowledge cameras, but no image pair reaches 0.9.
- Single stray matches (fathomnet-uiis10k, roboflow_fish-uiis10k).

When datasets are added or reprocessed, rerun both deduplication commands before merging.

## 3. Merging

`datasets/merge_all_datasets.py` hardlinks (no extra disk space) every image in `data/final/*` except those in `images_to_remove.csv` to `data/community_fish_detector/JPEGImages/`, and writes one COCO file with the two categories (fish, non-fish) and an `is_train` flag per image. It stops if `images_to_remove.csv` is missing or `data/community_fish_detector/` already exists.
