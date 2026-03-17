from pathlib import Path
import random

import supervision as sv

from aggregation_of_final_dataset.settings import Settings
from data_preview.visualize_roboflow_fish import (
    DATASET_SHORTNAME,
    download_data,
    join_all_images_and_annotations_into_single_coco_dataset,
)
from aggregation_of_final_dataset.utils import (
    split_coco_dataset_into_train_validation,
    compress_annotations_to_single_category,
    convert_coco_annotations_from_0_indexed_to_1_indexed,
)


settings = Settings()


def get_list_of_cameras_to_include_in_train_set(train_image_folder: Path) -> list[str]:
    """
    Using same split as provided by Roboflow
    """
    train_image_names = list(train_image_folder.glob("*.jpg"))
    train_image_names = [image_name.name for image_name in train_image_names]
    return train_image_names


def main():
    # 1. RAW
    # Download NOAA Data in Raw Directory
    raw_download_path = settings.raw_dir / DATASET_SHORTNAME
    raw_download_path.mkdir(parents=True, exist_ok=True)
    download_data(raw_download_path)

    # 2. PROCESSING
    # Create COCO Dataset
    processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
    processing_dir.mkdir(parents=True, exist_ok=True)

    coco_images_path = processing_dir / settings.images_folder_name
    coco_images_path.mkdir(parents=True, exist_ok=True)
    coco_annotations_path = processing_dir / settings.coco_file_name

    coco_images_path, coco_annotations_path = (
        join_all_images_and_annotations_into_single_coco_dataset(
            raw_download_path, coco_images_path, coco_annotations_path
        )
    )

    # Convert annotations to 1-indexed
    coco_annotations_path_1_indexed = processing_dir / "annotations_coco_1_indexed.json"
    coco_annotations_path = convert_coco_annotations_from_0_indexed_to_1_indexed(
        coco_annotations_path, coco_annotations_path_1_indexed
    )

    # Compress annotations to single category
    compressed_annotations_path = (
        processing_dir / "annotations_coco_compressed.json"
    )
    categories_filter = None
    compress_annotations_to_single_category(
        coco_annotations_path_1_indexed, categories_filter, compressed_annotations_path
    )

    # 3. FINAL
    # Build Logic to split into train and val based on camera name
    raw_train_set = raw_download_path / "train"
    train_image_names = get_list_of_cameras_to_include_in_train_set(raw_train_set)
    should_the_image_be_included_in_train_set = (
        lambda image_path: image_path in train_image_names
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
