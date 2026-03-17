from pathlib import Path
import random

import supervision as sv

from aggregation_of_final_dataset.settings import Settings
from data_preview.visualize_project_natick import (
    DATASET_SHORTNAME,
    download_data,
    add_extension_to_filename,
)
from aggregation_of_final_dataset.utils import (
    split_coco_dataset_into_train_validation,
    compress_annotations_to_single_category,
)


settings = Settings()


def get_list_of_cameras_to_include_in_train_set(image_folder: Path) -> list[str]:
    # Split the images randomly as all are from the same camera, location and datetime
    all_images = list(image_folder.glob("*.jpg"))
    train_ratio = 1 - settings.train_val_split_ratio
    train_size = int(len(all_images) * train_ratio)
    train_images = random.sample(all_images, train_size)
    train_images = [image.name for image in train_images]
    return train_images


def main():
    # 1. RAW
    # Download NOAA Data in Raw Directory
    raw_download_path = settings.raw_dir / DATASET_SHORTNAME
    raw_download_path.mkdir(parents=True, exist_ok=True)
    download_data(raw_download_path)

    # 2. PROCESSING
    processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
    processing_dir.mkdir(parents=True, exist_ok=True)

    # Create COCO Dataset (from Pascal VOC)
    dataset_path = raw_download_path / "data_release" / "fish_arrow_worms_annotation"
    images_path = dataset_path / "JPEGImages"
    annotations_path = dataset_path / "Annotations"

    # Create Valid VOC Annotations
    add_extension_to_filename(annotations_path)

    # Create COCO Dataset
    dataset = sv.DetectionDataset.from_pascal_voc(
        images_directory_path=str(images_path),
        annotations_directory_path=str(annotations_path),
    )
    coco_images_path = processing_dir / settings.images_folder_name
    coco_images_path.mkdir(parents=True, exist_ok=True)
    coco_annotations_path = processing_dir / settings.coco_file_name
    dataset.as_coco(str(coco_images_path), str(coco_annotations_path))

    # Compress annotations to single category
    compressed_annotations_path = (
        processing_dir / "project_natick_compressed_annotations.json"
    )
    categories_to_keep = ["Fish", "Squid"]
    compress_annotations_to_single_category(
        coco_annotations_path, categories_to_keep, compressed_annotations_path
    )

    # 3. FINAL
    # Build Logic to split into train and val based on camera name
    train_image_names = get_list_of_cameras_to_include_in_train_set(coco_images_path)
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
