# Utility API Reference

This file documents all shared utilities available in `datasets/utils/` for use in dataset scripts.

## Download Utilities (`datasets.utils.download`)

### `download_file(url: str, save_path: Path)`
Downloads a file from a URL with progress bar. Uses streaming with 1MB chunks.

### `extract_downloaded_file(download_path: Path, extract_to: Path, compression_type: CompressionType)`
Extracts a compressed file (ZIP or TAR). Deletes the compressed file after extraction.

### `download_and_extract(data_dir: Path, data_url: str, dataset_shortname: str, compression_type: CompressionType)`
Convenience function: downloads to `data_dir/<shortname>.<ext>` then extracts to `data_dir/`.

### `CompressionType` (Enum)
- `CompressionType.ZIP` — for .zip files
- `CompressionType.TAR` — for .tar files

## COCO Utilities (`datasets.utils.coco`)

### `compress_annotations_to_single_category(annotations_path: Path, categories_filter: Optional[List[str]], output_path: Path) -> Path`
Compresses all annotations to single "fish" category (id=1).
- If `categories_filter` is a list of strings, only annotations whose category name is in the list are kept.
- If `categories_filter` is `None`, ALL annotations are kept (all categories assumed to be fish).
- Writes result to `output_path`. Returns `output_path`.
- **Important**: Assumes category IDs are 1-indexed and match list indices (category_id - 1 = index in categories list).

### `convert_coco_annotations_from_0_indexed_to_1_indexed(input_path: Path, output_path: Path) -> Path`
Increments all category IDs and annotation category_ids by 1. Use when source dataset uses 0-indexed categories.

## Image Utilities (`datasets.utils.images`)

### `add_dataset_shortname_prefix_to_image_names(images_path: Path, annotations_path: Path, dataset_shortname: str)`
Renames all images to `<shortname>_<original_name>` and updates the COCO annotations file to match. Skips images already prefixed. **Must be called after annotations are finalized.**

### `copy_images_to_processing(dataset_shortname: str, source_dir: Path) -> Path`
Copies all image files from `source_dir` to `settings.intermediate_dir / shortname / JPEGImages/`. Returns the destination path.

### `remove_dataset_shortname_prefix_from_image_filename(image_filename: str, dataset_shortname: str) -> str`
Removes the prefix from a filename. Useful for extracting original identifiers for split logic.

## Split Utilities (`datasets.utils.split`)

### `split_coco_dataset_into_train_validation(source_images_path, source_annotations_path, train_dataset_path, val_dataset_path, get_split_for_image: Callable[[str], bool])`
Splits a COCO dataset into train and validation sets.
- `get_split_for_image`: function that takes an image filename and returns `True` if the image belongs in the train set.
- Creates `JPEGImages/` and `annotations_coco.json` inside both `train_dataset_path` and `val_dataset_path`.
- Both output directories must already exist.

### `get_train_images_with_random_splitting(images_path: Path) -> list[str]`
Returns list of image filenames for the train set using random 80/20 split. Uses `settings.random_state` for reproducibility.

## Visualization Utilities (`datasets.utils.visualization`)

### `save_preview_image(images_path: Path, annotations_path: Path, shortname: str)`
Loads the dataset with supervision, visualizes random samples with bounding boxes, and saves a sample image to `previews/<shortname>_sample_image.png`.

## Settings (`datasets.settings.Settings`)

Key settings used in scripts:
```python
settings = Settings()

settings.raw_dir           
settings.processed_dir     
settings.intermediate_dir  
settings.preview_dir       # <repo>/previews/

settings.images_folder_name    # "JPEGImages"
settings.coco_file_name        # "annotations_coco.json"
settings.coco_category_id      # 1
settings.coco_categories       # [{"id": 1, "name": "fish"}]

settings.train_dataset_suffix  # "_train"
settings.val_dataset_suffix    # "_val"
settings.train_val_split_ratio # 0.2
settings.random_state          # 42
```

## COCO Format Reference

Standard COCO annotation structure:
```json
{
  "images": [
    {"id": 1, "file_name": "shortname_image001.jpg", "width": 1920, "height": 1080}
  ],
  "annotations": [
    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [x, y, width, height], "area": 1234, "iscrowd": 0}
  ],
  "categories": [
    {"id": 1, "name": "fish"}
  ]
}
```

- `bbox`: `[x_top_left, y_top_left, width, height]`
- `area`: `width * height`
- `iscrowd`: always 0 for this project
- All IDs must be 1-indexed
