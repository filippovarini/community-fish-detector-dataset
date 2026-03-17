from typing import Set
from pathlib import Path

from sklearn.model_selection import train_test_split

from data_preview.visualize_brackish import (
    DATASET_SHORTNAME,
    download_data,
    join_all_images_and_annotations_into_single_coco_dataset,
)
from aggregation_of_final_dataset.utils import (
    convert_coco_annotations_from_0_indexed_to_1_indexed,
    compress_annotations_to_single_category,
    split_coco_dataset_into_train_validation,
    add_dataset_shortname_prefix_to_image_names,
    remove_dataset_shortname_prefix_from_image_filename,
)
from aggregation_of_final_dataset.settings import Settings


settings = Settings()


def get_unique_deployments(image_folder: Path) -> Set:
    deployments = set()
    for image_path in image_folder.glob("*.jpg"):
        image_name_without_dataset_prefix = (
            remove_dataset_shortname_prefix_from_image_filename(
                image_path.stem, DATASET_SHORTNAME
            )
        )
        deployment = "-".join(
            image_name_without_dataset_prefix.split("_jpg")[0].split("-")[:-1]
        )
        deployments.add(deployment)

    return deployments


def get_list_of_cameras_to_include_in_train_set(image_folder: Path) -> list[str]:
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
    processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
    processing_dir.mkdir(parents=True, exist_ok=True)

    coco_images_path = processing_dir / settings.images_folder_name
    coco_images_path.mkdir(parents=True, exist_ok=True)
    coco_annotations_path = processing_dir / settings.coco_file_name

    # Create combined dataset
    join_all_images_and_annotations_into_single_coco_dataset(
        raw_download_path, coco_images_path, coco_annotations_path
    )

    # The COCO categories are 0-indexed. We need to convert them to 1-indexed
    # for consistency with the other datasets
    coco_annotations_path_1_indexed = processing_dir / "annotations_coco_1_indexed.json"
    convert_coco_annotations_from_0_indexed_to_1_indexed(
        coco_annotations_path, coco_annotations_path_1_indexed
    )

    # Compress annotations to single category
    compressed_annotations_path = processing_dir / "annotations_coco_compressed.json"
    categories_filter = ["small_fish", "fish"]
    compress_annotations_to_single_category(
        coco_annotations_path_1_indexed, categories_filter, compressed_annotations_path
    )

    # We add the dataset shortname prefix to the image names to ensure
    # that the image names are unique across datasets
    add_dataset_shortname_prefix_to_image_names(
        coco_images_path,
        compressed_annotations_path,
        DATASET_SHORTNAME,
    )

    # 3. FINAL
    # Build Logic to split into train and val based on camera name
    train_deployments = get_list_of_cameras_to_include_in_train_set(coco_images_path)
    should_the_image_be_included_in_train_set = lambda image_filename: any(
        remove_dataset_shortname_prefix_from_image_filename(
            image_filename, DATASET_SHORTNAME
        ).startswith(deployment)
        for deployment in train_deployments
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
