from pathlib import Path
from typing import Set

from sklearn.model_selection import train_test_split

from aggregation_of_final_dataset.settings import Settings
from data_preview.visualize_deepfish import (
    DATASET_SHORTNAME,
    download_data,
    create_coco_dataset,
)
from aggregation_of_final_dataset.utils import (
    split_coco_dataset_into_train_validation,
    compress_annotations_to_single_category,
    add_dataset_shortname_prefix_to_image_names,
    remove_dataset_shortname_prefix_from_image_filename,
)


settings = Settings()


def get_unique_deployments(image_folder: Path) -> Set:
    deployments = set()
    for image_path in image_folder.glob("*.jpg"):
        image_filename_without_prefix = remove_dataset_shortname_prefix_from_image_filename(
            image_path.stem, DATASET_SHORTNAME
        )
        deployments.add(image_filename_without_prefix.split("_")[0])
    return deployments


def get_list_of_deployments_to_include_in_train_set(image_folder: Path) -> list[str]:
    # Split the camera names into train and val
    deployments = list(get_unique_deployments(image_folder))
    train_deployments, _ = train_test_split(
        list(deployments),
        test_size=settings.train_val_split_ratio,
        random_state=settings.random_state,
    )
    return train_deployments


def main():
    # 1. RAW
    # Download NOAA Data in Raw Directory
    raw_download_path = settings.raw_dir / DATASET_SHORTNAME
    raw_download_path.mkdir(parents=True, exist_ok=True)
    download_data(raw_download_path)

    # 2. PROCESSING
    # Build COCO Dataset
    processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
    processing_dir.mkdir(parents=True, exist_ok=True)

    coco_annotations_path = processing_dir / settings.coco_file_name
    images_path, annotations_path = create_coco_dataset(
        raw_download_path, coco_annotations_path
    )

    compressed_annotations_path = processing_dir / "annotations_coco_compressed.json"
    # Keep all categories as all are fish
    categories_filter = None
    compress_annotations_to_single_category(
        annotations_path, categories_filter, compressed_annotations_path
    )
    
    add_dataset_shortname_prefix_to_image_names(
        images_path=images_path,
        annotations_path=compressed_annotations_path,
        dataset_shortname=DATASET_SHORTNAME,
    )

    # 3. FINAL
    # Build Logic to split into train and val based on camera name
    train_deployments = get_list_of_deployments_to_include_in_train_set(images_path)
    should_the_image_be_included_in_train_set = (
        lambda image_filename: remove_dataset_shortname_prefix_from_image_filename(
            image_filename, DATASET_SHORTNAME
        ).split("_")[0] in train_deployments
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
        images_path,
        compressed_annotations_path,
        train_dataset_path,
        val_dataset_path,
        should_the_image_be_included_in_train_set,
    )


if __name__ == "__main__":
    main()
