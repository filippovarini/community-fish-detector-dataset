import os
import shutil
from pathlib import Path

import tqdm

from aggregation_of_final_dataset.settings import Settings
from aggregation_of_final_dataset.utils import (
    compress_annotations_to_single_category,
    split_coco_dataset_into_train_validation,
    add_dataset_shortname_prefix_to_image_names,
    remove_dataset_shortname_prefix_from_image_filename,
)
from data_preview.visualize_zebrafish import (
    DATASET_SHORTNAME,
    download_data,
    clean_annotations_and_get_df,
    dataframe_to_coco,
)


settings = Settings()


def main():
    # 1. RAW
    # Kagglehub downloads by default to  ~/.cache/kagglehub/
    # Download NOAA Data in Raw Directory
    raw_download_path = settings.raw_dir / DATASET_SHORTNAME
    raw_download_path.mkdir(parents=True, exist_ok=True)
    os.environ["KAGGLEHUB_CACHE"] = str(raw_download_path)
    download_data(raw_download_path)

    # 2. PROCESSING
    processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
    processing_dir.mkdir(parents=True, exist_ok=True)

    coco_images_path = processing_dir / settings.images_folder_name
    coco_images_path.mkdir(parents=True, exist_ok=True)
    coco_annotations_path = processing_dir / settings.coco_file_name

    # Clean and process annotations
    data_df, raw_images_path = clean_annotations_and_get_df(raw_download_path)
    dataframe_to_coco(data_df, coco_annotations_path)

    # Copy all raw images in coco images path
    images_generator = raw_images_path.glob("*.png")
    total_images = list(images_generator)
    for image_path in tqdm.tqdm(total_images, total=len(total_images)):
        shutil.copy2(image_path, coco_images_path / image_path.name)

    # Compress annotations to single category
    compressed_annotations_path = processing_dir / "annotations_coco_compressed.json"
    # Keep all categories (just 1: Zebrafish)
    categories_filter = None
    compress_annotations_to_single_category(
        coco_annotations_path, categories_filter, compressed_annotations_path
    )

    # Add dataset shortname prefix to image names
    add_dataset_shortname_prefix_to_image_names(
        images_path=coco_images_path,
        annotations_path=compressed_annotations_path,
        dataset_shortname=DATASET_SHORTNAME,
    )

    # 3. FINAL
    # Build Logic to split into train and val based on camera name
    # Only 2 videos: Vid1, Vid2. Split one in train and one in validation.
    # Not the ideal split ratio, but we give priority to not polluting
    # training/val with images from the same video
    should_the_image_be_included_in_train_set = (
        lambda image_name: remove_dataset_shortname_prefix_from_image_filename(
            image_name, DATASET_SHORTNAME
        ).startswith("Vid1")
    )

    train_dataset_path = (
        settings.processed_dir / f"{DATASET_SHORTNAME}{settings.train_dataset_suffix}"
    )
    val_dataset_path = (
        settings.processed_dir / f"{DATASET_SHORTNAME}{settings.val_dataset_suffix}"
    )
    train_dataset_path.mkdir(parents=True)
    val_dataset_path.mkdir(parents=True)

    split_coco_dataset_into_train_validation(
        coco_images_path,
        compressed_annotations_path,
        train_dataset_path,
        val_dataset_path,
        should_the_image_be_included_in_train_set,
    )


if __name__ == "__main__":
    main()
